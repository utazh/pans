#!/usr/bin/env python3
"""Export saved complete plans for fixed-ID DIAGNOSTICS; never changes scoring cost."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contiguous_fuxian.sharedkv_v24.records import load
from contiguous_fuxian.sharedkv_v27.common import torch_new,json_new
p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--task',required=True);p.add_argument('--role',choices=['fit','validation'],required=True);p.add_argument('--output-dir',required=True)
a=p.parse_args();out=Path(a.output_dir)
if out.exists():raise FileExistsError(out)
rows=load(a.records,a.task,a.role)
for h,r in rows.items():torch_new(out/(h+'.pt'),{'selected_by_layer':r['selection_by_layer']})
json_new(out/'scope.json',{'query_hashes':sorted(rows),'role':a.role,'timing_valid':False,'note':'Selector still executes when IDs replayed. Evaluation-query plans should instead use original --logits captures.'})
