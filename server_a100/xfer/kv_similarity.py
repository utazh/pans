"""观察 3 的实测：相邻层前缀 KV 有多不同，RAPS 的仿射读取映射能恢复多少。
(1) 全部相邻层对 (l-1, l)：逐 token、逐 KV head 的平均角距离 Ω/π 与相对误差 ||x_l - x_{l-1}|| / ||x_l||（K、V 分别）。
(2) 给定 reader checkpoint：每个消费层 ℓ，直接替代误差 ||X_ℓ - X_s||/||X_ℓ|| 与读取映射后误差 ||X_ℓ - (X_s A + a)||/||X_ℓ||（K 用 A/bk，V 用 B/bv）。
用法: kv_similarity.py <store_root> <task> <out_json> [reader.pt]"""
import json, sys
from pathlib import Path
import torch
store, task, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); ck = sys.argv[4] if len(sys.argv) > 4 else None
meta = json.loads((store / task / 'metadata.json').read_text()); n, h, d, L = meta['prefix_tokens'], meta['kv_heads'], meta['head_dim'], meta['layers']
def read(l, kind):
    raw = torch.frombuffer(bytearray((store / task / f'layer_{l:02d}_{kind}.bf16').read_bytes()), dtype=torch.uint8)
    return raw.view(torch.bfloat16).reshape(n, h, d).float().cuda()
res = dict(model_layers=L, kv_heads=h, prefix_tokens=n, adjacent={}, reader={})
for kind in ('key', 'value'):
    X = [read(l, kind) for l in range(L)]
    rows = []
    for l in range(1, L):
        a, b = X[l - 1], X[l]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).clamp(-1, 1)
        rows.append(dict(l=l, angular=(torch.arccos(cos) / torch.pi).mean().item(), cos=cos.mean().item(),
                         rel_err=((b - a).norm() / b.norm()).item()))
    res['adjacent'][kind] = rows
    if ck:
        bundle = torch.load(ck, map_location='cpu', weights_only=True); src = bundle['sources']
        for c, r in bundle['readers'].items():
            l = int(c); s = src[l]; Xs, Xl = X[s], X[l]
            A, bias = (r['A'], r['bk']) if kind == 'key' else (r['B'], r['bv'])
            A, bias = A.float().cuda(), bias.float().cuda()                     # [h, d, d], [h, d]
            mapped = torch.einsum('nhd,hde->nhe', Xs, A) + bias                 # 行向量重建 X_s A + a
            res['reader'].setdefault(kind, []).append(dict(consumer=l, source=s,
                direct_rel_err=((Xl - Xs).norm() / Xl.norm()).item(), mapped_rel_err=((Xl - mapped).norm() / Xl.norm()).item(),
                direct_cos=torch.nn.functional.cosine_similarity(Xs, Xl, dim=-1).mean().item(),
                mapped_cos=torch.nn.functional.cosine_similarity(mapped, Xl, dim=-1).mean().item()))
    del X; torch.cuda.empty_cache()
out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1))
for kind in ('key', 'value'):
    adj = res['adjacent'][kind]; deep = adj[L // 2:]
    msg = f"{task} L={L} {kind}: adjacent angular mean {sum(r['angular'] for r in adj)/len(adj):.3f} (deep half {sum(r['angular'] for r in deep)/len(deep):.3f}), rel_err {sum(r['rel_err'] for r in adj)/len(adj):.3f}"
    if ck and kind in res['reader']:
        rr = res['reader'][kind]
        msg += f" | reader: direct {sum(r['direct_rel_err'] for r in rr)/len(rr):.3f} -> mapped {sum(r['mapped_rel_err'] for r in rr)/len(rr):.3f}"
    print(msg, flush=True)
