#!/usr/bin/env python3
import json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'profiles/v21/viz_hybrid_shared_retry/viztracer.json').read_text())
events=[e for e in data['traceEvents'] if e['ph']=='X']
greedy=[e for e in events if e['name'].startswith('greedy_flexgen_completion ')];assert len(greedy)==8
decoders=sorted([e for e in events if e['name'].startswith('decoder_logits ') and '/sharedkv_v21/runtime.py:' in e['name']],key=lambda e:e['ts'])
scopes=[]
for e in greedy:
 scope=next(d for d in decoders if e['ts']<=d['ts'] and d['ts']+d['dur']<=e['ts']+e['dur']+1)
 scopes.append(scope)
totals=collections.defaultdict(lambda:[0.,0.,0.]);main=collections.defaultdict(lambda:[0.,0.,0.])
for scope in scopes:
 groups=collections.defaultdict(list)
 for e in events:
  if e['ts']>=scope['ts'] and e['ts']+e['dur']<=scope['ts']+scope['dur']+.01:groups[e['tid']].append(dict(e,self_us=e['dur']))
 for tid,es in groups.items():
  stack=[]
  for e in sorted(es,key=lambda x:(x['ts'],-x['dur'])):
   while stack and (e['ts']>=stack[-1]['ts']+stack[-1]['dur'] or e['ts']+e['dur']>stack[-1]['ts']+stack[-1]['dur']+.02):stack.pop()
   if stack:stack[-1]['self_us']-=e['dur']
   stack.append(e)
  for e in es:
   for d in ([totals,main] if tid==scope['tid'] else [totals]):
    d[e['name']][0]+=1;d[e['name']][1]+=e['dur']/1000;d[e['name']][2]+=max(0,e['self_us'])/1000
def rank(d,index):
 return [{'function':name,'calls_per_prefill':v[0]/8,'inclusive_ms_per_prefill':v[1]/8,'self_ms_per_prefill':v[2]/8} for name,v in sorted(d.items(),key=lambda kv:-kv[1][index])[:30]]
r={'scope':'Eight first prefill calls identified as the first v21 decoder inside each greedy completion; excludes label decoding and warmup. Self time subtracts traced direct children on the same thread, and includes untraced library/C calls and blocking waits. Threads overlap; sums are not a critical path. Profiled times must not be used as formal TTFT.','trace_events':len(events),'overflow':data['viztracer_metadata']['overflow'],'prefill_count':len(scopes),'mean_profiled_prefill_ms':sum(e['dur'] for e in scopes)/8000,'main_thread_by_self':rank(main,2),'main_thread_by_inclusive':rank(main,1),'all_threads_by_self':rank(totals,2)}
(ROOT/'results/v21_viz_summary.json').write_text(json.dumps(r,indent=2))
print('prefill',r['mean_profiled_prefill_ms'])
for row in r['main_thread_by_self'][:10]:print(row['function'],round(row['self_ms_per_prefill'],3))
