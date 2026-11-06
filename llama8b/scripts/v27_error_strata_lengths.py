
from pathlib import Path
import json,statistics
r=Path('/home/panzihang/src/prism_layerkv')
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained('/data1/llm/Qwen/Qwen2.5-7B-Instruct')
length_by_task={}
for task in ['sst2','subj','trec','rte']:
 folder='canonical_full_eval' if task=='subj' else 'full_eval'
 rs=map(json.loads,(r/f'data/v26/p0/{folder}/{task}.jsonl').read_text().splitlines())
 length_by_task[task]={x['uid']:len(tokenizer.encode(x['query_text'],add_special_tokens=False)) for x in rs}
assert length_by_task['trec']['trec-v26-raw-eval-0']==42
out=[]
for task in ['sst2','subj','trec','rte']:
 for budget in [5,10,25,50]:
  stem=f'v26_p0_{task}_b{budget}'+('_canonical' if task=='subj' else '')+'_quality_'
  a=list(map(json.loads,(r/f'results/{stem}I/scored_records.jsonl').read_text().splitlines()))
  b=list(map(json.loads,(r/f'results/{stem}S/scored_records.jsonl').read_text().splitlines()))
  assert [x['uid'] for x in a]==[x['uid'] for x in b]
  extra=r/f'results/{stem}I.extra.jsonl'
  lengths=[length_by_task[task][x['uid']] for x in a]
  if extra.exists():
   ext=list(map(json.loads,extra.read_text().splitlines()))
   if len(ext)==len(a):lengths=[x['query_tokens'] for x in ext]
  rows=[]
  for i,(x,y) in enumerate(zip(a,b)):
   vals=sorted(x['label_candidate_mean_logprobs'].values(),reverse=True)
   rows.append(dict(uid=x['uid'],correct=x['correct'],student=y['correct'],bad=x['correct'] and not y['correct'],good=not x['correct'] and y['correct'],margin=vals[0]-vals[1],query_tokens=lengths[i] if lengths else None,label=x['answer']))
  def summarize(rr):
   n=len(rr);return dict(n=n,bad=sum(x['bad'] for x in rr),good=sum(x['good'] for x in rr),delta_pp=100*sum(int(x['good'])-int(x['bad']) for x in rr)/max(n,1))
  bands={}
  for name,fn in [('margin_lt0.5',lambda x:x['margin']<.5),('margin_0.5_to2',lambda x:.5<=x['margin']<2),('margin_ge2',lambda x:x['margin']>=2)]:
   bands[name]=summarize([x for x in rows if fn(x)])
  if lengths:
   for name,fn in [('query_le64',lambda x:x['query_tokens']<=64),('query_65to128',lambda x:64<x['query_tokens']<=128),('query_gt128',lambda x:x['query_tokens']>128)]:
    bands[name]=summarize([x for x in rows if fn(x)])
  out.append(dict(task=task,budget=budget,overall=summarize(rows),strata=bands,classes={k:summarize([x for x in rows if x['label']==k]) for k in sorted({x['label'] for x in rows})}))
(r/'results/v27_v26_error_strata_with_lengths.json').write_text(json.dumps({'scope':'posthoc descriptive; no candidate selection or certification','cells':out},indent=2))
print('stratified',len(out),'cells')
