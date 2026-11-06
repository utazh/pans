#!/usr/bin/env python3
"""Install only new files into an audited v2 checkout. Never overwrite existing work."""
import argparse,hashlib,shutil,subprocess
from pathlib import Path
BASE='b48c45d779ecde8ae960a6e4837d8b6673c57600'
CHECKS={
 'scripts/run_sharedkv_v21.py':'2687662615672bece81e18b8dd957a8fc69f74ef',
'src/contiguous_fuxian/sharedkv_v21/runtime.py':'69e088baf05cc64009850a52a69792831123d0ca',
'src/contiguous_fuxian/sharedkv_v21/refine.py':'4093410f35cb0c1fd618c0d64eb0a7aad134bd7f',
'src/contiguous_fuxian/sharedkv_v21/ops.py':'e7b15b1407689b06673159d23774fd2c97e9ed64',
'src/contiguous_fuxian/sharedkv/core.py':'d83ff10040329bea613e1cd5c5c7b95c1e1e6ff7',
'src/contiguous_fuxian/sharedkv/runtime.py':'944ee1cd079cd75cc4e1d7b44000b6f1c679ad2c',
'src/contiguous_fuxian/sharedkv/io.py':'5cc0bc761955df308e3b352d62cf48c3660f3b05',
'src/contiguous_fuxian/sharedkv/fit.py':'4da7d62b41ef265631c311eea5a3f78d882fa478'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();repo=Path(a.repo).resolve();root=Path(__file__).resolve().parents[1]
    actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if actual!=BASE:raise RuntimeError(f'Use an isolated worktree at {BASE}; got {actual}')
    for name,expected in CHECKS.items():
        b=(repo/name).read_bytes();h=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if h!=expected:raise RuntimeError(f'Changed audited interface: {name}; do not bypass')
    moves=[]
    for folder in ('src','scripts','tests'):
        for source in (root/folder).rglob('*.py'):
            rel=source.relative_to(root);dst=repo/rel
            if dst.exists():raise FileExistsError(dst)
            moves.append((source,dst))
    docs=[(root/'README_ZH.md',repo/'README_HEADALIGN_V22.md')]
    docs += [(p,repo/'docs'/'headalign_v22'/p.name) for p in (root/'docs').glob('*.md')]
    for _,dst in docs:
        if dst.exists():raise FileExistsError(dst)
    for source,dst in moves+docs:
        print(f'{source} -> {dst}')
        if not a.dry_run:
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
if __name__=='__main__':main()
