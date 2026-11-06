#!/usr/bin/env python3
"""Create-only installation. No original source file is rewritten."""
from pathlib import Path
import argparse,ast,hashlib,json,shutil

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];dest=Path(a.repo).resolve(strict=True)
 manifest=json.loads((root/'MANIFEST.sha256.json').read_text())
 for name,expected in manifest.items():
  path=(root/name).resolve();path.relative_to(root)
  if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Package changed: {name}')
 guard=ast.parse((root/'src/contiguous_fuxian/sharedkv_v25/guard.py').read_text())
 checks=next(ast.literal_eval(n.value) for n in guard.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHECKS' for t in n.targets))
 for name,expected in checks.items():
  b=(dest/name).read_bytes();actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
  if actual!=expected:raise RuntimeError(f'Base interface mismatch: {name}; do not bypass')
 pairs=[]
 for name in manifest:
  rel=Path(name)
  if rel.parts[0] in ('src','scripts','tests'):
   pairs.append((root/rel,dest/rel))
  else:pairs.append((root/rel,dest/'docs/sharedkv_v25_package'/rel))
 for _,target in pairs:
  if target.exists():raise FileExistsError(f'Create-only installer refuses: {target}')
 for source,target in pairs:
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 print(f'Installed {len(pairs)} new files. Run v25 tests, then existing full CPU/CUDA and real-model parity tests.')
if __name__=='__main__':main()
