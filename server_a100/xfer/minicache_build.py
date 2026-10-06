"""MiniCache-Prefix 基线（Liu et al., NeurIPS'24）在前缀存储层面的实现，用于与 RAPS / KVSharer 做同口径质量对比。
按原文默认值：从 S=L/2 起，相邻层 (l-1, l) 两两合并（l = S+1, S+3, ...）；K、V 分别处理，逐 token、逐 KV head：
  Ω = arccos(cos(x^{l-1}, x^l)),  e = sin((1-t)Ω)/sinΩ · x^{l-1}/|x^{l-1}| + sin(tΩ)/sinΩ · x^l/|x^l|,  t = 0.6
  x̂^l = e · |x^l| / |e|（每层保留自己的模长）
  token 保留：每对层内角距离 d = Ω/π 最大的 5% (token, head) 保持原值不合并（原文 γ=0.05，约 5% token 保留）。
后缀/生成 token 的 KV 仍为原生（只改历史前缀）。选择索引沿用原生 K4 索引（与 KVSharer-Prefix 的适配口径相同）。
字节（解析值）：每对合并层存一份方向 + 两个模长；保留 token 两层全存。
用法: minicache_build.py <src_store> <dst_store> <dst_pcache> [--t 0.6 --retain 0.05]"""
import argparse, json, math, os, shutil, sys
from pathlib import Path
import torch
p = argparse.ArgumentParser(); p.add_argument('src'); p.add_argument('dst'); p.add_argument('pcache')
p.add_argument('--t', type=float, default=0.6); p.add_argument('--retain', type=float, default=0.05); a = p.parse_args()
src, dst = Path(a.src), Path(a.dst); dst.mkdir(parents=True, exist_ok=False)
TASKS = ['sst2', 'subj', 'trec', 'rte']; report = {}

def read(path, n, h, d):
    raw = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
    return raw.view(torch.bfloat16).reshape(n, h, d).float().cuda()
def write(path, x):
    path.write_bytes(x.to(torch.bfloat16).contiguous().view(torch.uint8).cpu().numpy().tobytes())

def merge(x0, x1, t, retain):
    n0, n1 = x0.norm(dim=-1, keepdim=True).clamp_min(1e-12), x1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    u0, u1 = x0 / n0, x1 / n1
    cos = (u0 * u1).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7); om = torch.arccos(cos); so = torch.sin(om)
    small = so < 1e-6
    w0 = torch.where(small, torch.full_like(om, 1 - t), torch.sin((1 - t) * om) / so)
    w1 = torch.where(small, torch.full_like(om, t), torch.sin(t * om) / so)
    e = w0 * u0 + w1 * u1; e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    r0, r1 = e * n0, e * n1
    d = (om / math.pi).squeeze(-1)                               # [tokens, heads]
    k = max(1, int(round(retain * d.numel())))
    thr = torch.topk(d.flatten(), k).values.min()
    keep = (d >= thr).unsqueeze(-1)
    return torch.where(keep, x0, r0), torch.where(keep, x1, r1), keep.float().mean().item(), d.mean().item()

for task in TASKS:
    meta = json.loads((src / task / 'metadata.json').read_text()); n, h, dd, L = meta['prefix_tokens'], meta['kv_heads'], meta['head_dim'], meta['layers']
    (dst / task).mkdir(); (dst / task / 'metadata.json').write_text(json.dumps(meta, indent=2))
    S = L // 2; pairs = [(l - 1, l) for l in range(S + 1, L, 2)]; merged = {x for pr in pairs for x in pr}
    stats = {}
    for l in range(L):
        if l not in merged:
            for kind in ('key', 'value'):
                shutil.copyfile(src / task / f'layer_{l:02d}_{kind}.bf16', dst / task / f'layer_{l:02d}_{kind}.bf16')
    for l0, l1 in pairs:
        for kind in ('key', 'value'):
            x0 = read(src / task / f'layer_{l0:02d}_{kind}.bf16', n, h, dd); x1 = read(src / task / f'layer_{l1:02d}_{kind}.bf16', n, h, dd)
            y0, y1, kept, dmean = merge(x0, x1, a.t, a.retain)
            write(dst / task / f'layer_{l0:02d}_{kind}.bf16', y0); write(dst / task / f'layer_{l1:02d}_{kind}.bf16', y1)
            stats[f'{l0}-{l1}_{kind}'] = dict(retained=kept, mean_angular_distance=dmean,
                                               rel_err_l0=((y0 - x0).norm() / x0.norm()).item(), rel_err_l1=((y1 - x1).norm() / x1.norm()).item())
    # 解析字节：合并对 = 一份方向(d) + 两个模长(2)（非保留 token）；保留 token 两层全存
    per_tok_layer = h * dd
    full = L * per_tok_layer
    mc = (L - len(merged)) * per_tok_layer + len(pairs) * ((1 - a.retain) * (per_tok_layer + 2 * h) + a.retain * 2 * per_tok_layer)
    report[task] = dict(layers=L, start=S, pairs=pairs, bytes_ratio_vs_full=mc / full, stats=stats)
    print(task, 'pairs', len(pairs), 'bytes ratio', round(mc / full, 4),
          'mean rel err K', round(sum(v['rel_err_l1'] for k, v in stats.items() if k.endswith('key')) / len(pairs), 4), flush=True)
(dst / 'minicache_report.json').write_text(json.dumps(dict(t=a.t, retain=a.retain, tasks=report), indent=1))
# 由变换后的前缀存储生成 pcache（与 v29_quality_pipeline 相同的构建与完成标记）
sys.path.insert(0, 'src')
from contiguous_fuxian.flexgen_pcache import FlexGenPcacheConfig, FlexGenPcacheStore
heads = tuple(int(x) for x in os.environ.get('PANS_SELECTOR_KV_HEAD_IDS', '0,1,2,3').split(','))
pc = FlexGenPcacheStore(FlexGenPcacheConfig(flexgen_root=Path('vendor/flexgen').resolve(), kv_dir=Path(a.pcache), chunk_size=16, gpu_cache_mb=0,
                                            cpu_cache_mb=0, cache_type='LRU', selector_kv_head_ids=heads))
for t in TASKS: pc.add_task(store_root=dst, task=t)
for f in Path(a.pcache).iterdir():
    if f.is_file():
        with f.open('rb') as fh: os.fsync(fh.fileno())
(Path(a.pcache) / '.contiguous_fuxian_complete').write_text(json.dumps(dict(schema_version=3, method='minicache_prefix_t0.6_retain0.05',
    chunk_size=16, selector_kv_head_ids=list(heads), online_selection=True, physical_layout='plain-logical-token-order',
    impress_reorder_sha256=None, registered_store_tasks=TASKS, build_report=str(dst / 'minicache_report.json')), indent=2))
print('MINICACHE_BUILD_DONE')
