#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    from contiguous_fuxian.sharedkv_v22.guard import check
    check(ROOT)
    from contiguous_fuxian.sharedkv_v22.calibrate import main
    main()
