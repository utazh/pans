#!/usr/bin/env python3
"""Predeclare a finite source-budget grid. No dev/test scores are inputs.

This does NOT invoke the old run_v24_ladder_and_eval certification stop rule.
Exploration runs may proceed with failed/unresolved NI, but all checkpoints
remain diagnostic. Do not retrofit a pass onto the prior failed protocol.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--teacher-fit',required=True);p.add_argument('--task',required=True)
 p.add_argument('--output-dir',required=True);p.add_argument('--protect-first',type=int,default=0)
 p.add_argument('--budgets',default='26,24,22,20,16,14');p.add_argument('--native-blocks',default='0,4')
 p.add_argument('--seeds',default='24');a=p.parse_args()
 from contiguous_fuxian.sharedkv_v25.guard import check
 from contiguous_fuxian.sharedkv_v24.records import load
 from contiguous_fuxian.sharedkv_v24.topology import build_stages
 from contiguous_fuxian.sharedkv_v25.policy import NativePolicy,payload_account
 check(ROOT);rows=load(a.teacher_fit,a.task,'fit');header=next(iter(rows.values()))
 groups,stages=build_stages(rows,protect_first=a.protect_first)
 budgets={int(x) for x in a.budgets.split(',')};repairs=[int(x) for x in a.native_blocks.split(',')]
 seeds=[int(x) for x in a.seeds.split(',')];out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
 jobs=[];available=set()
 for stage in stages:
  G=stage['unique_sources']
  if G not in budgets:continue
  available.add(G);path=out/f'g{G:02d}.json'
  stage.update(exploration_only=True,quality_certified=False,
               preregistration_note='Finite graph/repair-budget sweep; no certification results used to design topology')
  path.write_text(json.dumps(stage,indent=2))
  for r in repairs:
   pcy=NativePolicy(r)
   accounts=[payload_account(row['selection_by_layer'],stage['sources'],row['prefix_tokens'],
                 header['geometry']['kv_heads'],header['geometry']['head_dim'],pcy) for row in rows.values()]
   for seed in seeds:
    jobs.append(dict(plan=str(path),G=G,native_blocks=r,seed=seed,
        mean_selected_payload_reduction=sum(x['selected_payload_reduction_fraction'] for x in accounts)/len(accounts),
        status='planned_research_not_run',quality_certified=False))
 result=dict(groups=groups,jobs=jobs,unavailable_budgets=sorted(budgets-available),
             next_step='Train each listed candidate explicitly; no automatic pass, no final-test selection',
             inherited_NI='Previous failed gate unchanged. Preregister a separate independent final acceptance study.')
 (out/'study.json').write_text(json.dumps(result,indent=2));print(out/'study.json')
 if not jobs:raise ValueError('No legal requested topology')
if __name__=='__main__':main()
