#!/usr/bin/env python3
"""Add files after full preflight; never overwrite a different repository file."""
from pathlib import Path
import argparse,hashlib,json,sys
PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'src'))
from contiguous_fuxian.sharedkv_v26.guard import check

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);a=p.parse_args()
 repo=Path(a.repo).resolve(strict=True)
 manifest=json.loads((PACKAGE/'MANIFEST.sha256.json').read_text())
 for rel,expected in manifest.items():
  src=PACKAGE/rel
  if not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest()!=expected:
   raise ValueError(f'Package integrity mismatch: {rel}')
 check(repo)
 copies=[]
 for rel in manifest:
  parts=Path(rel).parts
  if parts[0] in ('src','scripts','tests'):
   dest=repo/rel
  elif parts[0]=='configs':
   dest=repo/'configs/v26'/Path(*parts[1:])
  elif parts[0]=='docs':
   dest=repo/'docs/v26'/Path(*parts[1:])
  elif rel in ('README_ZH.md','TEST_RESULTS.md','NOTICE.md'):
   dest=repo/'docs/v26'/rel
  else:continue
  b=(PACKAGE/rel).read_bytes()
  if dest.exists() and dest.read_bytes()!=b:
   raise FileExistsError(f'Refusing to overwrite: {dest}')
  copies.append((dest,b))
 # All conflicts checked before first write.
 for dest,b in copies:
  dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():
   with dest.open('xb') as f:f.write(b)
 print(json.dumps({'installed_or_identical_files':len(copies),'repo':str(repo),
                   'original_source_files_overwritten':0},indent=2))
if __name__=='__main__':main()
