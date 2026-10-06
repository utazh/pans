"""Deployable xKV-CLF build cost per prefix (GPU, torch.svd_lowrank as in the xKV implementation; q = r + 8, niter = 4):
read store K/V per 4-layer group, inverse-RoPE K, randomized low-rank factorization for each rank pair. The FP64 masters
used for numerical certification are reported separately (compile_seconds stored in the master files).
Usage: v29_xkv_buildcost.py   (GPU via CUDA_VISIBLE_DEVICES)"""
import json, time
from pathlib import Path
import torch
R = Path('/home/shensicheng/qyd/pans/src'); import sys; sys.path.insert(0, str(R / 'src'))
from contiguous_fuxian.sharedkv.io import read_prefix
from contiguous_fuxian.sharedkv_v27.baseline_math import rotary
out = {}
for task in ('sst2', 'subj', 'trec', 'rte'):
    meta = json.loads((R / f'runtime_data/prefix_store/{task}/metadata.json').read_text()); n = meta['prefix_tokens']
    pos = torch.arange(n, device='cuda'); res = {}
    for rk, rv in ((256, 384), (384, 576)):
        torch.cuda.synchronize(); t0 = time.perf_counter(); io = 0.0
        for lo in range(0, 28, 4):
            for kind, r in (('key', rk), ('value', rv)):
                t1 = time.perf_counter(); xs = []
                for lid in range(lo, lo + 4):
                    x, _ = read_prefix(R / 'runtime_data/prefix_store', task, lid, kind); x = x.to('cuda', torch.float32)
                    if kind == 'key': x = rotary(x, pos, theta=1e6, inverse=True)
                    xs.append(x.reshape(n, -1))
                torch.cuda.synchronize(); io += time.perf_counter() - t1
                X = torch.cat(xs, 1); torch.manual_seed(27)
                u, s, v = torch.svd_lowrank(X, q=min(min(X.shape), r + 8), niter=4)
                torch.cuda.synchronize()
        res[f'{rk}_{rv}'] = dict(total_s=time.perf_counter() - t0, read_and_rope_s=io)
    master = next((R / 'studies/v28_frozen/masters').glob(f'{task}_*.pt'))
    m = torch.load(master, map_location='cpu', weights_only=False)
    out[task] = dict(prefix_tokens=n, lowrank_gpu=res, fp64_master_certification_compile_s=m.get('compile_seconds'))
    print(task, json.dumps(out[task]), flush=True)
(R / 'results/v29_xkv_buildcost.json').write_text(json.dumps(out, indent=1))
