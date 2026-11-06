#!/usr/bin/env python3
"""Add files only; no source edits, deleted outputs, downloads or background jobs."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'src'))
from contiguous_fuxian.sharedkv_v27.guard import check
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);a=p.parse_args();root=Path(a.repo).resolve(strict=True)
manifest=json.loads((PACKAGE/'MANIFEST.sha256.json').read_text())
for name,h in manifest.items():
 raw=(PACKAGE/name).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=h:raise RuntimeError('Package checksum mismatch: '+name)
check(root)
items=[]
for folder in ('src','scripts','tests','configs'):
 for path in (PACKAGE/folder).rglob('*'):
  if path.is_file() and '__pycache__' not in path.parts:items.append((path,root/path.relative_to(PACKAGE)))
for folder,destination in [('docs','docs/sharedkv_v27'),('paper','paper/sharedkv_v27')]:
 for path in (PACKAGE/folder).rglob('*'):
  if path.is_file():items.append((path,root/destination/path.relative_to(PACKAGE/folder)))
for name in ('README_ZH.md','TEST_RESULTS.md','SCOPE.json'):
 items.append((PACKAGE/name,root/'docs/sharedkv_v27'/name))
conflicts=[str(dst) for _,dst in items if dst.exists()]
if conflicts:raise FileExistsError('Refusing overwrite:\n'+'\n'.join(conflicts))
for src,dst in items:
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
print('Added',len(items),'files. Base unchanged. GPU/model/SSD acceptance remains required.')
