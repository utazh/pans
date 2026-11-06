#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from contiguous_fuxian.sharedkv_v23.guard import check
from contiguous_fuxian.sharedkv_v23.train import main
if __name__=='__main__':
    check(ROOT);main()
