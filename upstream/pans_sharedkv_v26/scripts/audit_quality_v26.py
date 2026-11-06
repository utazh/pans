#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contiguous_fuxian.sharedkv_v26.analysis import records,paired_quality
p=argparse.ArgumentParser()
p.add_argument('--baseline',required=True);p.add_argument('--student',required=True)
p.add_argument('--margin',type=float,default=.01);p.add_argument('--alpha-per-route',type=float,default=.025)
p.add_argument('--output',required=True);a=p.parse_args()
r=paired_quality(records(a.baseline),records(a.student),margin=a.margin,alpha=a.alpha_per_route)
with Path(a.output).open('x',encoding='utf8') as f:json.dump(r,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps(r,indent=2,ensure_ascii=False))
