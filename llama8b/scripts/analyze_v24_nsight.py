#!/usr/bin/env python3
"""Hardware activity from Nsight, scoped to the exact TTFT NVTX ranges."""
import sqlite3,json,statistics,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def union_ms(intervals):
 intervals=sorted(intervals);total=0;lo=hi=None
 for x,y in intervals:
  if lo is None:lo,hi=x,y
  elif x>hi:total+=hi-lo;lo,hi=x,y
  else:hi=max(hi,y)
 return (total+(hi-lo if lo is not None else 0))/1e6
out={}
for path in sorted((ROOT/'profiles/v24').glob('nsys_*.sqlite')):
 c=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
 strings=dict(c.execute('select id,value from StringIds'))
 nvtx=[(s,e,text if text is not None else strings.get(tid,''),thread) for s,e,text,tid,thread in c.execute('select start,end,text,textId,globalTid from NVTX_EVENTS where end>start')]
 scopes=[x for x in nvtx if x[2]=='TTFT/logits_ready'];assert len(scopes)==8,(path,len(scopes))
 kern=[(s,e,strings[n]) for s,e,n in c.execute('select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL')]
 mem=list(c.execute('select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY'))
 api=[(s,e,strings[n],tid) for s,e,n,tid in c.execute('select start,end,nameId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME')]
 rows=[];all_names=collections.defaultdict(lambda:[0,0]);all_kern=collections.defaultdict(lambda:[0,0]);all_api=collections.defaultdict(lambda:[0,0])
 for start,end,_,main_tid in scopes:
  ks=[k for k in kern if start<=k[0]<end];ms=[m for m in mem if start<=m[0]<end]
  rows.append({'ttft_profiled_ms':(end-start)/1e6,'gpu_kernel_count':len(ks),'gpu_kernel_sum_ms':sum(e-s for s,e,_ in ks)/1e6,'gpu_busy_union_ms':union_ms([(max(s,start),min(e,end)) for s,e,_ in ks]+[(max(m[0],start),min(m[1],end)) for m in ms]),'h2d_bytes':sum(m[2] for m in ms if m[3]==1),'h2d_count':sum(m[3]==1 for m in ms),'h2d_sum_ms':sum(m[1]-m[0] for m in ms if m[3]==1)/1e6,'d2h_bytes':sum(m[2] for m in ms if m[3]==2),'d2d_bytes':sum(m[2] for m in ms if m[3]==8)})
  for s,e,name,tid in nvtx:
   if start<=s<end and e<=end:
    group='sharedkv/resolve_all_layers' if name.startswith('sharedkv/resolve_') else ('sharedkv/attention_all_layers' if name.startswith('sharedkv/attention_') else name)
    all_names[group][0]+=1;all_names[group][1]+=(e-s)/1e6
  for s,e,name in ks:all_kern[name][0]+=1;all_kern[name][1]+=(e-s)/1e6
  for s,e,name,tid in api:
   if tid==main_tid and start<=s<end:all_api[name][0]+=1;all_api[name][1]+=(e-s)/1e6
 def summary(d):
  return [{'name':name,'count_per_ttft':values[0]/len(scopes),'inclusive_ms_per_ttft':values[1]/len(scopes)} for name,values in sorted(d.items(),key=lambda kv:-kv[1][1])]
 out[path.stem]={'n':len(scopes),'means':{k:statistics.mean(r[k] for r in rows) for k in rows[0]},'requests':rows,'nvtx_inclusive':summary(all_names),'kernel_summary':summary(all_kern)[:30],'main_thread_cuda_api':summary(all_api)[:20]}
 print(path.stem,out[path.stem]['means'])
c=None
result={'scope':'Nsight CUPTI hardware copies/kernels whose GPU start is within the exact TTFT/logits_ready NVTX range (ends after CUDA synchronization). Excludes model loading/warmup/label scoring. H2D includes payload, selector index, metadata; it is not payload-only. Kernel/API/NVTX summed durations can overlap or nest; do not sum them into a critical path. Profiled times are diagnostic, not formal performance. Other experimental process may share host during capture.','variants':out}
(ROOT/'results/v24_nsight_summary.json').write_text(json.dumps(result,indent=2))
