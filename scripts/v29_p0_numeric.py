"""v29 P0 numeric check of the mechanism ablations at one layer, plus label-level comparisons.
capture <out.pt> <run_evidence_v26 args...> : run the unchanged chain, saving the first 3 consumer-layer attention
    inputs (q, source pk/pv, private tail tk/tv) and the packed reader of the first request.
analyze <capture.pt>                           : FP64 equivalence of read-side vs explicit reconstruction and of unit-reader
    vs direct sharing; then the error of the actual BF16 implementations of each path against the FP64 reference.
labels                                          : full candidate-score diff, label-distribution KL, margin change for the
    fixed-plan runs in studies/v29/p1ab_validate_fixed (explicit vs read-side; direct vs unit reader)."""
import json, math, runpy, sys
from pathlib import Path
ROOT = Path('/home/panzihang/src/prism_layerkv'); sys.path.insert(0, str(ROOT / 'src'))
import torch
MODE = sys.argv[1]

if MODE == 'capture':
    out = Path(sys.argv[2]); del sys.argv[1:3]
    import contiguous_fuxian.sharedkv_v21.ops as ops
    orig, saved = ops.dispatch, []
    def dispatch(q, pk, pv, tk, tv, *, reader=None, packed=None, **kw):
        if reader is not None and packed is not None and len(saved) < 3:
            saved.append(dict(q=q.detach().cpu(), pk=pk.detach().cpu(), pv=pv.detach().cpu(), tk=tk.detach().cpu(), tv=tv.detach().cpu(),
                              A_t=packed.A_transposed.cpu(), B=packed.B.cpu(), bk=packed.bk_float.cpu(), bv=packed.bv.cpu()))
            if len(saved) == 3: torch.save(saved, out)
        return orig(q, pk, pv, tk, tv, reader=reader, packed=packed, **kw)
    ops.dispatch = dispatch
    sys.argv = [str(ROOT / 'scripts/run_evidence_v26.py')] + sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name='__main__')

elif MODE == 'analyze':
    calls = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
    def attn(q, k, v, causal):  # q [t,hq,d], k/v [n,hk,d]; bottom-right causal; returns out [t,hq,d], lse [hq,t]
        t, hq, d = q.shape; n, hk, _ = k.shape; g = hq // hk
        kk, vv = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
        s = torch.einsum('thd,nhd->htn', q, kk) * d ** -0.5
        if causal:
            mask = torch.arange(n)[None, :] > (torch.arange(t)[:, None] + (n - t))
            s = s.masked_fill(mask[None], float('-inf'))
        lse = torch.logsumexp(s, -1)
        return torch.einsum('htn,nhd->thd', torch.softmax(s, -1), vv), lse
    def read_side(q, pk, pv, tk, tv, A, B, bk, bv):  # two segments + LSE merge (the runtime algebra)
        t, hq, d = q.shape; hk = A.shape[0]; g = hq // hk
        qh = q.reshape(t, hk, g, d)
        qr = torch.einsum('thgj,hij->thgi', qh, A).reshape(t, hq, d)            # Q A^T
        off = (torch.einsum('thgj,hj->thg', qh, bk).reshape(t, hq) * d ** -0.5).T  # q.bk/sqrt(d)  [hq,t]
        up, lp = attn(qr, pk, pv, False); ut, lt = attn(q, tk, tv, True)
        y = torch.einsum('thgi,hij->thgj', up.reshape(t, hk, g, d), B).reshape(t, hq, d) + bv.repeat_interleave(g, 0)[None]
        lp = lp + off; z = torch.logaddexp(lp, lt); beta = (lp - z).exp().T[..., None]
        return beta * y + (1 - beta) * ut
    def explicit(q, pk, pv, tk, tv, A, B, bk, bv):
        kt = torch.einsum('nhd,hde->nhe', pk, A) + bk[None]; vt = torch.einsum('nhd,hde->nhe', pv, B) + bv[None]
        return attn(q, torch.cat([kt, tk]), torch.cat([vt, tv]), True)[0]
    def direct(q, pk, pv, tk, tv):
        return attn(q, torch.cat([pk, tk]), torch.cat([pv, tv]), True)[0]
    rel = lambda a, b: float((a - b).norm() / b.norm())
    import contiguous_fuxian.sharedkv_v21.ops as ops
    from contiguous_fuxian.sharedkv.core import Reader
    res = []
    for i, c in enumerate(calls):
        q, pk, pv, tk, tv = (c[k][0] for k in ('q', 'pk', 'pv', 'tk', 'tv'))
        A = c['A_t'].transpose(-1, -2); B, bk, bv = c['B'], c['bk'], c['bv']
        f = lambda x: x.double()
        ref = read_side(f(q), f(pk), f(pv), f(tk), f(tv), f(A), f(B), f(bk), f(bv))
        exp64 = explicit(f(q), f(pk), f(pv), f(tk), f(tv), f(A), f(B), f(bk), f(bv))
        I = torch.eye(A.shape[-1], dtype=torch.float64).repeat(A.shape[0], 1, 1); z = torch.zeros_like(f(bk))
        unit64 = read_side(f(q), f(pk), f(pv), f(tk), f(tv), I, I, z, z); dir64 = direct(f(q), f(pk), f(pv), f(tk), f(tv))
        r = dict(call=i, n_prefix=pk.shape[0], t=q.shape[0], reader_dtype=str(c['A_t'].dtype),
                 fp64_readside_vs_explicit=rel(exp64, ref), fp64_unit_vs_direct=rel(dir64, unit64))
        if torch.cuda.is_available():  # actual BF16 runtime implementations vs FP64 reference
            dev = lambda x: x[None].cuda()
            packed = ops.PackedReader(c['A_t'].cuda(), B.cuda(), bk.cuda(), bv.cuda())
            o_read = ops.packed_attention(dev(q), dev(pk), dev(pv), dev(tk), dev(tv), packed, backend='vllm2')[0].double().cpu()
            Ac, Bc = A.cuda(), B.cuda()  # mirror scripts/v29_exec_variant.py exactly (compute in the reader dtype)
            kt = (torch.einsum('bnhd,hde->bnhe', dev(pk).to(Ac.dtype), Ac) + bk.cuda().to(Ac.dtype)[None, None]).to(pk.dtype)
            vt = (torch.einsum('bnhd,hde->bnhe', dev(pv).to(Bc.dtype), Bc) + bv.cuda().to(Bc.dtype)[None, None]).to(pv.dtype)
            o_exp = ops.native_attention(dev(q), kt, vt, dev(tk), dev(tv), backend='vllm2')[0].double().cpu()
            o_dir = ops.native_attention(dev(q), dev(pk), dev(pv), dev(tk), dev(tv), backend='vllm2')[0].double().cpu()
            Ic = torch.eye(A.shape[-1]).repeat(A.shape[0], 1, 1).to(c['A_t'].dtype).cuda()
            o_unit = ops.packed_attention(dev(q), dev(pk), dev(pv), dev(tk), dev(tv),
                                          ops.PackedReader(Ic, Ic.to(c['B'].dtype), torch.zeros_like(c['bk']).cuda(), torch.zeros_like(c['bv']).cuda()),
                                          backend='vllm2')[0].double().cpu()
            r.update(bf16_readside_vs_fp64=rel(o_read, ref), bf16_explicit_vs_fp64=rel(o_exp, ref), bf16_readside_vs_bf16_explicit=rel(o_read, o_exp),
                     bf16_unit_vs_fp64=rel(o_unit, unit64), bf16_direct_vs_fp64=rel(o_dir, unit64), bf16_unit_vs_bf16_direct=rel(o_unit, o_dir),
                     max_abs_read_vs_exp=float((o_read - o_exp).abs().max()), output_scale=float(ref.abs().max()))
        res.append(r); print(json.dumps(r), flush=True)
    Path(sys.argv[2]).with_suffix('.analysis.json').write_text(json.dumps(res, indent=1))

elif MODE == 'labels':
    O = ROOT / 'studies/v29/p1ab_validate_fixed'
    def recs(n): return {r['uid']: r for r in map(json.loads, (O / n / 'scored_records.jsonl').read_text().splitlines())}
    def dist(r):
        s = r['label_candidate_mean_logprobs']; ks = sorted(s); m = max(s.values()); e = [math.exp(s[k] - m) for k in ks]; z = sum(e)
        return [x / z for x in e], [s[k] for k in ks]
    def margin(sc):
        a = sorted(sc, reverse=True); return a[0] - a[1]
    out = {}
    for cell in ('trec_b25', 'sst2_b25'):
        for a, b in (('explicit', 'readside'), ('direct', 'unitreader')):
            A, Bv = recs(f'{cell}_{a}'), recs(f'{cell}_{b}')
            kl, dm, ds = [], [], []
            for u in A:
                pa, sa = dist(A[u]); pb, sb = dist(Bv[u])
                kl.append(sum(x * (math.log(x) - math.log(max(y, 1e-30))) for x, y in zip(pb, pa) if x > 0))
                dm.append(margin(sa) - margin(sb)); ds.append(max(abs(x - y) for x, y in zip(sa, sb)))
            out[f'{cell}:{a}_vs_{b}'] = dict(n=len(A), mean_kl=sum(kl) / len(kl), max_kl=max(kl), mean_abs_margin_change=sum(map(abs, dm)) / len(dm),
                                            max_abs_margin_change=max(map(abs, dm)), max_abs_score_diff=max(ds),
                                            min_teacher_margin=min(margin(dist(Bv[u])[1]) for u in A))
            print(f'{cell} {a} vs {b}:', json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out[f"{cell}:{a}_vs_{b}"].items()}))
    (ROOT / 'results/v29_p0_numeric_labels.json').write_text(json.dumps(out, indent=1))
