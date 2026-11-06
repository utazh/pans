#!/usr/bin/env python3
"""Install additive files into v1. No upstream or old v2 overwrites."""
import argparse,shutil,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from contiguous_fuxian.sharedkv_v21.guard import check
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True)
a=p.parse_args();repo=a.repo.resolve();check(repo)
jobs=[]
for folder in ('src','scripts','tests'):
    for src in (root/folder).rglob('*.py'):
        if src.name==Path(__file__).name:continue
        dst=repo/src.relative_to(root)
        if dst.exists() and dst.read_bytes()!=src.read_bytes():raise FileExistsError(f'Refusing overwrite: {dst}')
        jobs.append((src,dst))
for src,dst in jobs:
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():shutil.copy2(src,dst)
print(f'Installed/verified {len(jobs)} files. Published v1 remains unchanged.')
