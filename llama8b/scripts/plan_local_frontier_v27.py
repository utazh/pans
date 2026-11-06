#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from contiguous_fuxian.sharedkv_v24.records import load,FIELDS,PLAN_FORMAT,assert_plan
from contiguous_fuxian.sharedkv_v27.topology import bounded_plans
from contiguous_fuxian.sharedkv_v27.common import json_new
p=argparse.ArgumentParser();p.add_argument('--teacher-fit',required=True);p.add_argument('--task',required=True)
p.add_argument('--max-distance',type=int,default=1);p.add_argument('--max-consumers',type=int,default=2)
p.add_argument('--protect-first',type=int,default=4);p.add_argument('--budgets',default='16,20,24,28');p.add_argument('--output-dir',required=True)
a=p.parse_args();out=Path(a.output_dir)
if out.exists():raise FileExistsError(out)
r=load(a.teacher_fit,a.task,'fit');head=next(iter(r.values()))
found=bounded_plans(list(r.values()),head['geometry']['layers'],head['period'],max_distance=a.max_distance,max_consumers=a.max_consumers,protect_first=a.protect_first)
requested=sorted(set(map(int,a.budgets.split(','))));out.mkdir(parents=True)
for g in requested:
 if g not in found:continue
 item=found[g]
 for row in r.values():assert_plan(row,item['sources'])
 plan={'format':PLAN_FORMAT,**{k:head[k] for k in FIELDS},**item,'unique_sources':g,'design_query_hashes':sorted(r),
       'quality_certified':False,'status':'geometry_only_candidate','constraints':vars(a)}
 json_new(out/f'g{g}.json',plan)
json_new(out/'feasibility.json',{'requested':requested,'feasible':sorted(found),'infeasible_requested':[g for g in requested if g not in found],
 'warning':'No fallback widens distance or changes token IDs; infeasibility is a result, not an error to hide.'})
