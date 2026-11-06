#!/usr/bin/env python3
"""Check exact selection compatibility BEFORE fitting; never form a hidden union."""
from pathlib import Path
import argparse,json,sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from contiguous_fuxian.sharedkv_v23.contracts import Group
p=argparse.ArgumentParser();p.add_argument('--plans',required=True);p.add_argument('--start',type=int,required=True)
p.add_argument('--end',type=int,required=True);p.add_argument('--source',type=int,required=True)
p.add_argument('--native-layers',default='');p.add_argument('--period',type=int,default=8);p.add_argument('--output',required=True)
a=p.parse_args();group=Group(a.start,a.end,a.source,tuple(int(x) for x in a.native_layers.split(',') if x))
if Path(a.output).exists():raise FileExistsError(a.output)
rows=[]
for path in sorted(Path(a.plans).glob('*.pt')):
    x=torch.load(path,map_location='cpu',weights_only=True)
    plan=x.get('selection_by_layer',x.get('selected_by_layer'))
    if plan is None:raise ValueError('Not a selection-plan or full group trace directory')
    group.validate(len(plan),a.period)
    bad=[l for l in group.consumers if list(plan[l])!=list(plan[group.source])]
    rows.append({'query':path.stem,'incompatible_consumers':bad,'counts':{str(i):len(plan[i]) for i in range(a.start,a.end)}})
if not rows:raise ValueError('No selection plans')
report={'queries':len(rows),'all_compatible':all(not r['incompatible_consumers'] for r in rows),'rows':rows,
    'note':'A failed equality check is a contract failure, not model approximation error. Preserve exceptions or change selection in BOTH baselines explicitly.'}
Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
