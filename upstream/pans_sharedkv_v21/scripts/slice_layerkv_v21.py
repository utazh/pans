#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contiguous_fuxian.sharedkv.io import load_safe,save_new
from contiguous_fuxian.sharedkv_v21.bundles import isolate
p=argparse.ArgumentParser(description='Extract specified consumers for isolated diagnostic testing; never certifies quality')
p.add_argument('--input',required=True);p.add_argument('--output',required=True)
p.add_argument('--targets',required=True,help='Consumer layer numbers, comma-separated, 0-based')
p.add_argument('--direct',action='store_true');a=p.parse_args()
save_new(a.output,isolate(load_safe(a.input),a.targets.split(','),direct=a.direct))
