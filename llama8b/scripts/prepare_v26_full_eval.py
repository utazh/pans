
from v26_common import *
from contiguous_fuxian.paper_tasks import *
r=ROOT;raw=Path('/home/panzihang/src/contiguous_fuxian/data/paper_tasks')
norm=lambda x:' '.join(x.casefold().split())
report={}
for task in ['sst2','subj','trec','rte']:
 train,ev=load_task_rows(task,raw,42)
 for pref in ['p0','p1']:
  folder=r/f'data/v26/{pref}/full_eval';folder.mkdir(exist_ok=True)
  template=json.loads((r/f'data/v26/{pref}/eval/{task}.jsonl').read_text().splitlines()[0])
  demos={norm(format_example(task,x,False)) for x in train if norm(format_example(task,x,True)) in norm(template['prefix_text'])}
  cal={norm(json.loads(line)['query_text']) for split in ['fit','val'] for line in (r/f'data/v26/{pref}/{split}/{task}.jsonl').read_text().splitlines()}
  rows=[];seen=set();clean=[];duplicate=[];overlap=[]
  for i,x in enumerate(ev):
   q=format_example(task,x,False)+QWEN_ASSISTANT_PREFIX;uid=f'{task}-v26-raw-eval-{i}';key=norm(q)
   assert key not in cal
   row=dict(template,uid=uid,query_text=q,answer=x['label'])
   rows.append(row)
   if key in seen:duplicate.append(uid)
   elif norm(format_example(task,x,False)) in demos:overlap.append(uid)
   else:clean.append(uid)
   seen.add(key)
  with (folder/f'{task}.jsonl').open('x') as f:f.write(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
  report[f'{pref}-{task}']=dict(full_count=len(rows),deduplicated_demo_disjoint_uids=clean,duplicate_uids=duplicate,prefix_demo_overlap_uids=overlap,calibration_disjoint=True)
with (r/'audit/v26_full_eval_membership.json').open('x') as f:json.dump(report,f,indent=2)
print({k:{'full':v['full_count'],'clean':len(v['deduplicated_demo_disjoint_uids'])} for k,v in report.items()})
