#!/usr/bin/env python3
"""Additive installation. Does not fetch models, alter data or modify upstream."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from contiguous_fuxian.sharedkv_v23.guard import check
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);a=p.parse_args()
repo=Path(a.repo).resolve();check(repo)
manifest=json.loads((ROOT/'MANIFEST.sha256.json').read_text())
for name,sha in manifest.items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError(f'Package hash mismatch: {name}')
files=[]
for folder in ('src','scripts','tests','configs'):
    for src in (ROOT/folder).rglob('*'):
        if not src.is_file() or '__pycache__' in src.parts:continue
        dst=repo/src.relative_to(ROOT)
        if dst.exists():raise FileExistsError(dst)
        files.append((src,dst))
for src,dst in files:
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
print(f'Installed {len(files)} new files; upstream unchanged. Run new + existing tests before GPU experiments.')
