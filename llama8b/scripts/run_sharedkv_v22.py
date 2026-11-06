#!/usr/bin/env python3
"""All CLI options are the published run_sharedkv_v21.py options. Adds mapped KV-head reading."""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    from contiguous_fuxian.sharedkv_v22.guard import check
    check(ROOT)
    import contiguous_fuxian.sharedkv_v21.runtime as v21
    from contiguous_fuxian.sharedkv_v22.runtime import Runtime
    previous=v21.Runtime
    v21.Runtime=Runtime
    try:
        sys.argv[0]=str(ROOT/'scripts/run_sharedkv_v21.py')
        runpy.run_path(sys.argv[0],run_name='__main__')
    finally:v21.Runtime=previous
if __name__=='__main__':main()
