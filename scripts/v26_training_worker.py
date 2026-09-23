
from v26_common import *
import argparse,torch
from contiguous_fuxian.sharedkv_v24.records import load,FIELDS,PLAN_FORMAT,assert_plan
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=2);p.add_argument('--prefix',default='p0');p.add_argument('--tasks',default='sst2,subj,rte,trec');p.add_argument('--budgets',default='5,10,25,50');a=p.parse_args()
def graph(row):
 counts=[len(x) for x in row['selection_by_layer']];sources=list(range(28))
 # Preserve frozen historical G16 when geometrically compatible.
 old=json.loads((ROOT/'configs/v25_explore/g16.json').read_text())['sources']
 if all(row['selection_by_layer'][i]==row['selection_by_layer'][s] for i,s in enumerate(old)):return old
 # Prefix/budget geometry-only mapping, no scores/labels/quality used.
 need=12
 for start in [24,16,8,0]:
  remaining=list(range(start+1,min(start+8,28)))
  groups={}
  for i in remaining:groups.setdefault(tuple(row['selection_by_layer'][i]),[]).append(i)
  for group in sorted(groups.values(),key=lambda g:(-len(g),-max(g))):
   if need<=0:break
   n=min(need,len(group)-1)
   if n<=0:continue
   chosen=group[-(n+1):];src=chosen[len(chosen)//2]
   for i in chosen:sources[i]=src
   need-=n
 assert need==0
 return sources
for task in a.tasks.split(','):
 for budget in map(int,a.budgets.split(',')):
  tag=f'v26_{a.prefix}_{task}_b{budget}';fit=ROOT/f'runtime_data/{tag}_teacher_fit';val=ROOT/f'runtime_data/{tag}_teacher_val'
  if a.prefix=='p0' and task=='trec' and budget==25:
   print('FROZEN HISTORICAL',tag,flush=True);continue
  for split,role,folder in [('fit','fit',fit),('val','validation',val)]:
   name=tag+'_capture_'+split;base=argv(name,task=task,budget=budget,gpu=a.gpu,n=64,diagnostic=True,prefix=a.prefix,split=split)
   cmd=[PY,'-u','scripts/collect_teacher_v26.py','--teacher-records',str(folder),'--split',role,'--',*base[3:]]
   run(name,cmd,a.gpu)
  f=load(fit,task,'fit');v=load(val,task,'validation');row=next(iter(f.values()));sources=graph(row)
  for item in list(f.values())+list(v.values()):assert_plan(item,sources)
  plan=dict(format=PLAN_FORMAT,**{k:row[k] for k in FIELDS},sources=sources,unique_sources=len(set(sources)),design_query_hashes=sorted(f),status='fixed_G16_geometry_only_no_quality_search',quality_certified=False)
  path=ROOT/f'configs/{tag}_plan.json'
  with path.open('x') as h:json.dump(plan,h,indent=2)
  store=ROOT/'runtime_data/prefix_store' if a.prefix=='p0' else ROOT/'runtime_data/v26_prefix_p1'
  cmd=[PY,'-u','scripts/train_instrumented_v25.py','--native-blocks','4','--worklog',str(ROOT/f'audit/{tag}_compute.json'),'--','--model-path',MODEL,'--store-root',str(store),'--task',task,'--teacher-fit',str(fit),'--teacher-val',str(val),'--plan',str(path),'--steps','200','--eval-every','25','--seed','24','--device','cuda','--output',str(ROOT/f'assets/{tag}/{task}.pt')]
  run(tag+'_train',cmd,a.gpu)
