"""Materialize missing IMPRESS selector-key (head_token) files <prefix>_<layer>_None_k.npy in the v29 c64 Pcache.
Method is validated first: every EXISTING head file is recomputed from the prefix store + reorder manifest and
must match byte-for-byte; only then are missing files written. Usage: v29_fill_impress_head.py [--write]"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch
R = Path('/home/shensicheng/qyd/pans/src'); K = R / 'runtime_data/v29_pcache_impress_c64_ssd'
man = json.loads((R / 'configs/v29_impress_reorder_p0_fit4.json').read_text())
order = ['sst2', 'subj', 'trec', 'rte']  # --store-tasks order == prefix id
sys.path.insert(0, str(R / 'src'))
from contiguous_fuxian.sparse_qwen_reprefill import read_store_info

def perm_for(task, layer):
    return man['tasks'][task]['physical_to_logical'][layer]

def expected(pid, layer):
    task = order[pid]
    raw = np.fromfile(R / f'runtime_data/prefix_store/{task}/layer_{layer:02d}_key.bf16', dtype=np.uint16)
    meta = json.loads((R / f'runtime_data/prefix_store/{task}/metadata.json').read_text())
    n, h, d = meta['prefix_tokens'], meta['kv_heads'], meta['head_dim']
    key = torch.from_numpy(raw.astype(np.int16)).view(torch.bfloat16).reshape(n, h, d)
    p = torch.tensor(perm_for(task, layer), dtype=torch.long)
    return key[p][:, [0], :].to(torch.float16).numpy()

def same(arr, path):  # Pcache token files are raw headerless np.memmap buffers
    return Path(path).read_bytes() == np.ascontiguousarray(arr).tobytes()

print('manifest task entry type:', type(man['tasks']['sst2']).__name__,
      list(man['tasks']['sst2'])[:5] if isinstance(man['tasks']['sst2'], dict) else len(man['tasks']['sst2']))
have = sorted(K.glob('*_None_k.npy')); checked = ok = 0
for f in have:
    pid, layer = map(int, f.name.split('_')[:2]); checked += 1
    ok += same(expected(pid, layer), f)
print(f'validation: {ok}/{checked} existing head files reproduced exactly')
missing = [(p, l) for p in range(4) for l in range(28) if not (K / f'{p}_{l}_None_k.npy').exists()]
print('missing', missing)
if '--write' in sys.argv:
    assert ok == checked and checked > 0, 'method not validated; refusing to write'
    written = []
    for p, l in missing:
        arr = expected(p, l); path = K / f'{p}_{l}_None_k.npy'
        with path.open('xb') as fh: fh.write(np.ascontiguousarray(arr).tobytes())
        assert same(arr, path); written.append((p, l, hashlib.sha256(path.read_bytes()).hexdigest()))
    (R / 'audit/v29_impress_head_fill.json').write_text(json.dumps(dict(validated=f'{ok}/{checked}', written=written), indent=1))
    print('WROTE', len(written))
