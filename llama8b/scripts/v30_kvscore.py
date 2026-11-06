"""v30 baselines CAKE and LAVa adapted to SSD prefix re-prefill: attention statistics on Qwen2.5-7B.

Reference implementations (read, not imported):
  CAKE (ICLR 2025, github.com/antgroup/cakekv @ c4675ce, cake/model/modify_qwen2.py):
    window A = softmax(q[-w:] k^T / sqrt d) per query head, w = 32
    layer preference P_l = H^(1/tau1) * V^(1/tau2), H = -sum A log A over A[:, -w:, :-w], V = sum_j Var_rows(A[:, -w:, j])
    token indicator I = mean_rows(A) + gamma * var_rows(A) (gamma = 200), avg_pool1d(k=5), mean over the query heads of a KV group
  LAVa (arXiv 2509.09754, EMNLP Findings 2025, Eq. 5-7):
    s_{l,h}[i] = (max_k ||V_{l,h}[k]||_1 / w) * sum_{window rows} A_h[i]; max-pool k=7; GQA: max over heads in a group
    layer budget  B_l ∝ e_l = -sum_{h,i} s^ log s^ / (H N),  s^ = s / sum_{h,i} s

Two observation windows (modes):
  prefix : the last w=32 tokens of the shared prefix, at prefix registration time (query-independent; deployable).
  oracle : the request's own suffix tokens (last <= 32) attending over the FULL prefix KV (needs every prefix key; quality
           reference only, never a deployable latency point).
Adaptation to the block-granular native-KV engine (stated, not hidden): a selection unit is a 16-token chunk of all KV heads of
one layer, so per-token scores are summed over KV groups (all heads of a token are restored together) and over the tokens of
a chunk. Per-head dynamic budgets of LAVa therefore become per-layer budgets; the prefix-mode window chunks are always kept.

Outputs runtime_data/v30_kvscore/<mode>_<task>.npz with per-layer chunk scores and layer statistics.
Usage: v30_kvscore.py prefix <task> | oracle <task> <bundle_dir>   (CUDA_VISIBLE_DEVICES selects the GPU)"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import os
R = Path(os.environ.get('V30_ROOT', '/home/panzihang/src/prism_layerkv')); sys.path.insert(0, str(R / 'src'))
MODEL = os.environ.get('V30_MODEL', '/data1/llm/Qwen/Qwen2.5-7B-Instruct'); W = 32; CHUNK = 16; GAMMA = 200.0
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from contiguous_fuxian.sparse_qwen_reprefill import _token_hash
from contiguous_fuxian.sharedkv.io import query_hash

mode, task = sys.argv[1], sys.argv[2]
import os
# QuALITY reuse: V30_STORE (prefix store root), V30_BUNDLE (bundle dir for the prefix text), V30_TAG (output name prefix)
STORE = Path(os.environ.get('V30_STORE', R / 'runtime_data/prefix_store')); TAG = os.environ.get('V30_TAG', '')
OUT = R / 'runtime_data/v30_kvscore'; OUT.mkdir(parents=True, exist_ok=True)
dst = OUT / f'{mode}_{TAG}{task}.npz'
if dst.exists(): raise SystemExit(f'exists: {dst}')
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
cfg = model.config; L, H, HKV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
D = cfg.hidden_size // H; G = H // HKV
meta = json.loads((STORE / task / 'metadata.json').read_text())
bundle = Path(sys.argv[3]) if mode == 'oracle' else Path(os.environ.get('V30_BUNDLE', R / f'studies/v28_frozen/bundles/{task}_b25/eval'))
rows = [json.loads(l) for l in (bundle / f'{task}.jsonl').read_text().splitlines()]
prefix_text = rows[0]['prefix_text']; assert all(r['prefix_text'] == prefix_text for r in rows)
pids = tok(prefix_text, add_special_tokens=False).input_ids
N = len(pids)
assert N == meta['prefix_tokens'] and _token_hash(pids) == meta['token_hash'], 'prefix identity mismatch'
NB = (N + CHUNK - 1) // CHUNK

captured = {}
def pre_hook(lid):
    def f(mod, args, kwargs):
        h = kwargs['hidden_states']; cos, sin = kwargs['position_embeddings']
        t = h.shape[1]; take = slice(max(0, t - captured['rows']), t)
        q = mod.q_proj(h[:, take]).view(1, -1, H, D).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, take], sin[:, take])
        captured[lid] = q[0].float()                      # [H, rows, D]
    return f
for lid, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_pre_hook(pre_hook(lid), with_kwargs=True)

def layer_kv(cache, lid):
    lay = cache.layers[lid] if hasattr(cache, 'layers') else None
    k = lay.keys if lay is not None else cache.key_cache[lid]
    v = lay.values if lay is not None else cache.value_cache[lid]
    return k[0].float(), v[0].float()                     # [HKV, T, D]

def stats(q, k, v, n_cols_scored, row_pos):
    """q [H,w,D] at absolute positions row_pos; k,v [HKV,T,D]. Returns CAKE and LAVa quantities over the first
    n_cols_scored columns (the evictable region)."""
    kk = k.repeat_interleave(G, 0); T = kk.shape[1]
    s = torch.einsum('hwd,htd->hwt', q, kk) / D ** 0.5
    cols = torch.arange(T, device=q.device)
    s = s.masked_fill(cols[None, None, :] > row_pos[None, :, None], float('-inf'))
    a = torch.softmax(s, -1)                              # [H,w,T] fp32
    reg = a[:, :, :n_cols_scored]
    disp = float(-(reg * torch.log(reg + 1e-10)).sum())
    var_sum = float(torch.var(reg, dim=1).sum())
    cake = reg.mean(1) + GAMMA * torch.var(reg, dim=1)    # [H, n]
    cake = F.avg_pool1d(cake[:, None], kernel_size=5, padding=2, stride=1)[:, 0]
    cake_grp = cake.view(HKV, G, -1).mean(1)              # CAKE: mean over the query heads of a KV group
    vmax = v.abs().sum(-1).amax(1).repeat_interleave(G, 0)   # max_k ||V_h[k]||_1 over all cached positions
    lava = vmax[:, None] / q.shape[1] * reg.sum(1)        # [H, n]
    lava = F.max_pool1d(lava[:, None], kernel_size=7, padding=3, stride=1)[:, 0]
    lava_grp = lava.view(HKV, G, -1).amax(1)              # LAVa GQA: max over heads in a group
    sh = lava_grp / lava_grp.sum()
    ent = float(-(sh * torch.log(sh + 1e-30)).sum() / (HKV * n_cols_scored))
    return disp, var_sum, cake_grp.sum(0), lava_grp.sum(0), ent    # token scores summed over KV groups

def to_chunks(x, n):
    out = torch.zeros(NB, device=x.device); idx = torch.arange(n, device=x.device) // CHUNK
    out.index_add_(0, idx, x[:n]); return out

@torch.no_grad()
def run_prefix():
    captured['rows'] = W
    ids = torch.tensor([pids], device='cuda')
    out = model(input_ids=ids, use_cache=True); cache = out.past_key_values
    k0, _ = layer_kv(cache, 0)
    store_k0 = np.fromfile(STORE / task / 'layer_00_key.bf16', dtype=np.uint16).reshape(N, HKV, D)
    store_k0 = torch.from_numpy(store_k0.astype(np.int32) << 16).view(torch.float32).cuda().permute(1, 0, 2)
    rel = float((k0 - store_k0).norm() / store_k0.norm())
    pos = torch.arange(N - W, N, device='cuda'); res = dict(disp=[], var=[], cake=[], lava=[], ent=[])
    for lid in range(L):
        k, v = layer_kv(cache, lid)
        d_, v_, c_, l_, e_ = stats(captured[lid], k, v, N - W, pos)
        res['disp'].append(d_); res['var'].append(v_); res['ent'].append(e_)
        res['cake'].append(to_chunks(c_, N - W).cpu().numpy()); res['lava'].append(to_chunks(l_, N - W).cpu().numpy())
    forced = sorted({t // CHUNK for t in range(N - W, N)})
    np.savez(dst, mode='prefix', task=task, N=N, NB=NB, window=W, forced_chunks=np.array(forced),
             disp=np.array(res['disp']), var=np.array(res['var']), lava_entropy=np.array(res['ent']),
             cake_chunk=np.stack(res['cake']), lava_chunk=np.stack(res['lava']), layer0_key_rel_err=rel)
    print(json.dumps(dict(task=task, N=N, NB=NB, forced=forced, layer0_key_rel_err_vs_store=rel)), flush=True)

@torch.no_grad()
def run_oracle():
    ids = torch.tensor([pids], device='cuda'); captured['rows'] = 1
    cache = model(input_ids=ids, use_cache=True).past_key_values
    acc = dict(disp=[], var=[], ent=[], cake=[], lava=[], qhash=[], uid=[], qlen=[]); t0 = time.time()
    for i, r in enumerate(rows):
        qids = tok(str(r['query_text']), add_special_tokens=False).input_ids; Ls = len(qids); w = min(W, Ls)
        captured['rows'] = w
        model(input_ids=torch.tensor([qids], device='cuda'), past_key_values=cache, use_cache=True,
              position_ids=torch.arange(N, N + Ls, device='cuda')[None], cache_position=torch.arange(N, N + Ls, device='cuda'))
        pos = torch.arange(N + Ls - w, N + Ls, device='cuda'); d, va, en, cc, ll = [], [], [], [], []
        for lid in range(L):
            k, v = layer_kv(cache, lid)
            d_, v_, c_, l_, e_ = stats(captured[lid], k, v, N, pos)   # evictable region = the whole prefix
            d.append(d_); va.append(v_); en.append(e_)
            cc.append(to_chunks(c_, N).cpu().numpy()); ll.append(to_chunks(l_, N).cpu().numpy())
        cache.crop(N)
        acc['disp'].append(d); acc['var'].append(va); acc['ent'].append(en); acc['cake'].append(np.stack(cc)); acc['lava'].append(np.stack(ll))
        acc['qhash'].append(query_hash(qids)); acc['uid'].append(r['uid']); acc['qlen'].append(Ls)
        if i % 100 == 0: print(i, len(rows), round(time.time() - t0, 1), flush=True)
    np.savez(dst, mode='oracle', task=task, N=N, NB=NB, window=W, bundle=str(bundle), forced_chunks=np.array([], dtype=int),
             disp=np.array(acc['disp']), var=np.array(acc['var']), lava_entropy=np.array(acc['ent']),
             cake_chunk=np.stack(acc['cake']).astype(np.float32), lava_chunk=np.stack(acc['lava']).astype(np.float32),
             qhash=np.array(acc['qhash']), uid=np.array(acc['uid']), qlen=np.array(acc['qlen']))
    print('ORACLE_DONE', task, len(rows), flush=True)

run_prefix() if mode == 'prefix' else run_oracle()
