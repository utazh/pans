"""通用版 v29_fill_impress_head：IMPRESS c64 Pcache 的选择器 key 文件 <pid>_<layer>_None_k.npy（原始无头 memmap，
FP16，形状 [tokens, 1, d]）= 前缀存储 key 按 manifest 重排后取 KV 头 0。先逐字节校验全部已有文件，全部一致后才补写缺失文件。
用法: fill_impress_head.py <root> <c64_dir> <manifest> [--write]"""
import json, sys
from pathlib import Path
import numpy as np, torch
R, K, M = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()); write = '--write' in sys.argv
order = ['sst2', 'subj', 'trec', 'rte']
def expected(pid, layer):
    task = order[pid]; meta = json.loads((R / f'runtime_data/prefix_store/{task}/metadata.json').read_text())
    n, h, d = meta['prefix_tokens'], meta['kv_heads'], meta['head_dim']
    raw = np.fromfile(R / f'runtime_data/prefix_store/{task}/layer_{layer:02d}_key.bf16', dtype=np.uint16)
    key = torch.from_numpy(raw.astype(np.int16)).view(torch.bfloat16).reshape(n, h, d)
    p = torch.tensor(M['tasks'][task]['physical_to_logical'][layer], dtype=torch.long)
    return np.ascontiguousarray(key[p][:, [0], :].to(torch.float16).numpy())
L = M['tasks']['sst2']['layers']; checked = ok = missing = 0
for pid in range(4):
    for layer in range(L):
        f = K / f'{pid}_{layer}_None_k.npy'
        if f.exists():
            checked += 1; ok += f.read_bytes() == expected(pid, layer).tobytes()
        else:
            missing += 1
print(f'existing checked {checked}, byte-identical {ok}, missing {missing}', flush=True)
if checked and ok != checked: raise SystemExit('existing head files do not match the method; not writing')
if write and missing:
    for pid in range(4):
        for layer in range(L):
            f = K / f'{pid}_{layer}_None_k.npy'
            if not f.exists(): f.write_bytes(expected(pid, layer).tobytes())
    print('written', missing)
print('FILL_DONE')
