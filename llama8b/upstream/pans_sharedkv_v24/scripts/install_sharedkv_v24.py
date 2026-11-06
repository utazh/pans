#!/usr/bin/env python3
"""Install additive files into an independently checked published-v2.3 tree."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);a=p.parse_args()
    target=Path(a.repo).resolve(strict=True)
    manifest=json.loads((ROOT/'MANIFEST.sha256.json').read_text())
    for name,expected in manifest.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('Unsafe package path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Package changed: '+name)
    path=ROOT/'src/contiguous_fuxian/sharedkv_v24/guard.py'
    spec=importlib.util.spec_from_file_location('v24_guard',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.check(target)
    operations=[]
    for name in manifest:
        path=Path(name)
        if path.parts[0] in ('src','scripts','tests','configs'):
            dest=target/path
        elif path.parts[0]=='docs':
            dest=target/'docs/sharedkv_v24'/Path(*path.parts[1:])
        elif name in ('README_ZH.md','TEST_RESULTS.md'):
            dest=target/'docs/sharedkv_v24'/name
        else:continue
        if dest.exists():raise FileExistsError(f'Additive installation refuses overwrite: {dest}')
        operations.append((ROOT/name,dest))
    for src,dest in operations:
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    print(json.dumps({'installed':len(operations),'base':m.BASE,'repo':str(target),'original_files_overwritten':0},indent=2))
if __name__=='__main__':main()
