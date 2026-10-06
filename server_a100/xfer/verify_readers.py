"""解压 3090 reader 包后校验：(1) 全部 .pt 先用 weights_only=True 加载（只允许张量/基本容器，拒绝任意 pickle 对象）；
(2) audit/v29_final_selection.json 中每格选中 checkpoint 的 SHA256；(3) v29 candidate manifest 中 G16/G20/G24 的 SHA256（若记录）。"""
import hashlib, json, sys
from pathlib import Path
import torch
S = Path('/home/shensicheng/qyd/pans/src'); OLD = '/home/panzihang/src/prism_layerkv'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pts = sorted(S.glob('studies/**/readers/**/*.pt'))
for p in pts:
    torch.load(p, map_location='cpu', weights_only=True)
print('weights_only 安全加载通过:', len(pts))
fin = json.loads((S / 'docs/layerkv_v4/audit/v29_final_selection.json').read_text())['cells']
bad = 0
for cell, c in fin.items():
    if c['mode'] != 'shared': print(f'{cell:9s} {c["config"]:4s} (I, 无 checkpoint)'); continue
    d = Path(c['adapter_dir'].replace(OLD, str(S))); pt = sorted(d.glob('*.pt'))
    ok = bool(pt) and sha(pt[0]) == c['checkpoint_sha256']; bad += not ok
    print(f'{cell:9s} {c["config"]:4s} {"OK" if ok else "MISMATCH"} {pt[0].relative_to(S) if pt else "missing"}')
man = S / 'studies/v29/readers/manifest.json'
if man.exists():
    m = json.loads(man.read_text())['cells']; n = 0
    for cell, arms in m.items():
        for arm, info in arms.items():
            h = info.get('sha256') or info.get('checkpoint_sha256')
            if not h: continue
            d = Path(info['adapter_dir'].replace(OLD, str(S))); pt = sorted(d.glob('*.pt'))
            ok = bool(pt) and sha(pt[0]) == h; n += 1; bad += not ok
            if not ok: print('manifest MISMATCH', cell, arm)
    print('manifest 校验条目:', n)
print('ALL_OK' if bad == 0 else f'BAD={bad}')
