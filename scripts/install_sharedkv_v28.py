#!/usr/bin/env python3
"""Install ONLY new v28 files into the audited v2.7 worktree; refuse overwrites."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',required=True)
    args=parser.parse_args();repo=Path(args.repo).resolve(strict=True)
    from contiguous_fuxian.sharedkv_v28.common import check,sha
    check(repo)
    inventory=json.loads((ROOT/'MANIFEST.json').read_text())
    for name,digest in inventory['files'].items():
        source=ROOT/name
        if sha(source)!=digest:raise ValueError('Package changed: '+name)
    files=[]
    for sub in ('src','scripts','tests'):
        for source in sorted((ROOT/sub).rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts:
                files.append((source,repo/source.relative_to(ROOT)))
    for source in sorted((ROOT/'docs').glob('*.md')):
        files.append((source,repo/'docs/sharedkv_v28'/source.name))
    for name in ('README_ZH.md','TEST_RESULTS.md'):
        files.append((ROOT/name,repo/'docs/sharedkv_v28'/name))
    for source,destination in files:
        if destination.exists():raise FileExistsError('Refusing overwrite: '+str(destination))
    for source,destination in files:
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
    print(json.dumps({'installed_files':len(files),'repo':str(repo),'old_files_overwritten':0},indent=2))

if __name__=='__main__':main()
