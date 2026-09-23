#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contiguous_fuxian.sharedkv_v26.study import build
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True)
a=p.parse_args();r=build(json.loads(Path(a.spec).read_text()),a.output)
print(json.dumps({'jobs':len(r['jobs']),'unique_queries':r['n_unique_queries'],'manifest':str(Path(a.output)/'study.json')}))
