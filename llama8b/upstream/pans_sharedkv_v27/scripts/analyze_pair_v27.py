#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contiguous_fuxian.sharedkv_v27.quality import load_records,paired_quality
from contiguous_fuxian.sharedkv_v27.common import json_new
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--student',required=True)
p.add_argument('--alpha',type=float,default=.05);p.add_argument('--output',required=True)
a=p.parse_args();out=paired_quality(load_records(a.baseline),load_records(a.student),a.alpha);json_new(a.output,out);print(out)
