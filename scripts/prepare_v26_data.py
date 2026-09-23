
from v26_common import *
import random,re,shutil
from transformers import AutoTokenizer
from contiguous_fuxian.paper_tasks import *
out=ROOT/'data/v26';out.mkdir(exist_ok=False)
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
count=lambda x:len(tok(x,add_special_tokens=False).input_ids)
norm=lambda x:' '.join(x.casefold().split())
raw=Path('/home/panzihang/src/contiguous_fuxian/data/paper_tasks')
audit={}
for task in ['sst2','subj','trec','rte']:
 train,ev=load_task_rows(task,raw,42)
 old=[json.loads(x) for x in (ROOT/f'data/paper_task_bundles_full_eval/{task}.jsonl').read_text().splitlines()]
 prefix0=old[0]['prefix_text'];labels=old[0]['labels']
 key=lambda row:norm(format_example(task,row,False))
 evkeys={key(x) for x in ev}
 # Check demo membership from complete formatted examples, not substring sentence matches.
 p0demos={key(x) for x in train if norm(format_example(task,x,True)) in norm(prefix0)}
 assert len(p0demos)>=PAPER_TASKS[task].fewshot_examples-3,(task,len(p0demos))
 seen=set();eligible=[]
 for row in train:
  k=key(row)
  if k in seen or k in evkeys or k in p0demos:continue
  seen.add(k);eligible.append(row)
 if task=='trec':
  fitold=[json.loads(x) for x in (ROOT/'data/v24_splits/fit/trec.jsonl').read_text().splitlines()]
  valold=[json.loads(x) for x in (ROOT/'data/v24_splits/val/trec.jsonl').read_text().splitlines()]
  calkeys={norm(x['query_text'].split(QWEN_ASSISTANT_PREFIX)[0]) for x in fitold+valold}
 else:calkeys=set()
 demos=length_matched_stratified_sample(task,[x for x in eligible if key(x) not in calkeys],PAPER_TASKS[task].fewshot_examples,PAPER_TASKS[task].paper_prefix_tokens,count,2602)
 demokeys={key(x) for x in demos}
 assert not demokeys & (p0demos|evkeys|calkeys)
 prefix1=QWEN_USER_PREFIX+task_header(task)+''.join(format_example(task,x,True) for x in demos)
 pool=[x for x in eligible if key(x) not in demokeys and count(format_example(task,x,False)+QWEN_ASSISTANT_PREFIX)<240]
 random.Random(2603).shuffle(pool)
 if task!='trec':assert len(pool)>=128
 def rec(row,uid,prefix):
  return dict(uid=uid,task=task,prefix_text=prefix,query_text=format_example(task,row,False)+QWEN_ASSISTANT_PREFIX,answer=row['label'],labels=labels,label_continuation_prefix='')
 for name,prefix in [('p0',prefix0),('p1',prefix1)]:
  evalrows=[dict(x,prefix_text=prefix,label_continuation_prefix='') for x in old]
  if task=='trec':
   fit=[dict(x,prefix_text=prefix) for x in fitold];val=[dict(x,prefix_text=prefix) for x in valold]
  else:
   fit=[rec(x,f'{task}-v26-fit-{i}',prefix) for i,x in enumerate(pool[:64])]
   val=[rec(x,f'{task}-v26-val-{i}',prefix) for i,x in enumerate(pool[64:128])]
  for split,rows in [('eval',evalrows),('fit',fit),('val',val)]:
   folder=out/name/split;folder.mkdir(parents=True,exist_ok=True)
   with (folder/f'{task}.jsonl').open('x') as f:f.write(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
  sets=[{norm(x['query_text']) for x in rows} for rows in [evalrows,fit,val]]
  assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
  # Preserve complete historical evaluation, explicitly count duplicate and demo overlaps.
  audit[f'{task}-{name}']=dict(prefix_tokens=count(prefix),eval_count=len(evalrows),eval_unique=len(sets[0]),fit=len(fit),val=len(val),prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),eval_demo_overlap=sum(norm(x['query_text'].split(QWEN_ASSISTANT_PREFIX)[0]) in (p0demos if name=='p0' else demokeys) for x in evalrows),fresh_holdout=False)
 print(task,audit[f'{task}-p0'],audit[f'{task}-p1'],flush=True)
with (ROOT/'audit/v26_data.json').open('x') as f:json.dump(audit,f,indent=2)
