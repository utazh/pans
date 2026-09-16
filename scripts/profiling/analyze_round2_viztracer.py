import json,collections,argparse
from pathlib import Path
r=Path('/home/panzihang/src/prism_pans')
p=argparse.ArgumentParser();p.add_argument('profile_dir',type=Path);a=p.parse_args();base=a.profile_dir.resolve()
data=json.loads((base/'viztracer_run/viztracer.json').read_text())
events=[e for e in data['traceEvents'] if e.get('ph')=='X' and e.get('dur',0)>0]
roots=sorted([e for e in events if e['name'].startswith('greedy_flexgen_completion (')],key=lambda e:e['ts'])
assert len(roots)==4,(len(roots),len(events))
windows=[]
for root in roots:
 decoders=[e for e in events if e['name'].startswith('flexgen_sparse_decoder_logits (') and root['ts']<=e['ts']<root['ts']+root['dur']]
 first=min(decoders,key=lambda e:e['ts'])
 windows.append((first['ts'],first['ts']+first['dur']))
main_tid=roots[0]['tid']
totals=collections.defaultdict(lambda:[0.,0.,0])
threads=collections.Counter()
for a,b in windows:
 bythread=collections.defaultdict(list)
 for event in events:
  if event['ts']<b and event['ts']+event['dur']>a:
   e=dict(event);e['start']=max(a,e['ts']);e['end']=min(b,e['ts']+e['dur']);e['_self']=e['end']-e['start']
   bythread[e['tid']].append(e);threads[e['tid']]+=1
 for tid,es in bythread.items():
  es.sort(key=lambda e:(e['start'],-e['end']))
  stack=[]
  for e in es:
   while stack and e['start']>=stack[-1]['end']:stack.pop()
   if stack:stack[-1]['_self']-=max(0,min(e['end'],stack[-1]['end'])-e['start'])
   stack.append(e)
  for e in es:
   if str(r/'src') not in e['name'] and str(r/'vendor') not in e['name']:continue
   key=('main' if tid==main_tid else 'worker',e['name'])
   v=totals[key];v[0]+=e['end']-e['start'];v[1]+=max(0,e['_self']);v[2]+=1
ranked=[dict(thread=k[0],name=k[1],inclusive_ms_per_request=v[0]/4000,exclusive_traced_ms_per_request=v[1]/4000,calls_per_request=v[2]/4) for k,v in totals.items()]
ranked.sort(key=lambda x:-x['exclusive_traced_ms_per_request'])
out=dict(metadata=data.get('viztracer_metadata'),recorded_threads=dict(threads),prefill_windows_us=windows,hotspots=ranked)

# CPU timeline intersections are elapsed intervals, not additive CPU utilization.
def union(spans):
    out=[]
    for a,b in sorted(spans):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((a,b))
    return out
def duration(spans):return sum(b-a for a,b in union(spans))/1000
def intersection(left,right):
    return union([(max(a,c),min(b,d)) for a,b in union(left) for c,d in union(right) if max(a,c)<min(b,d)])
overlaps=[]
for a,b in windows:
    pref=[];wait=[];mainwork=[]
    for e in events:
        left=max(a,e['ts']);right=min(b,e['ts']+e['dur'])
        if right<=left:continue
        if e['tid']!=main_tid and e['name'].startswith('_run_prefetch ('):pref.append((left,right))
        if e['tid']==main_tid and e['name'].startswith('_resolve_entry ('):wait.append((left,right))
        if e['tid']==main_tid and any(e['name'].startswith(n+' (') for n in ['configure_online_layer_selection','_layer_causal_mask','_cache_layout']):mainwork.append((left,right))
    overlaps.append(dict(prefill_wall_ms=(b-a)/1000,worker_prefetch_union_ms=duration(pref),main_payload_resolve_entry_union_ms=duration(wait),worker_overlap_main_resolve_entry_ms=duration(intersection(pref,wait)),worker_overlap_main_selection_mask_layout_ms=duration(intersection(pref,mainwork))))
out['cpu_elapsed_overlap']=overlaps

(base/'viztracer_worker_breakdown.json').write_text(json.dumps(out,indent=2))
assert any(tid!=main_tid for tid in threads),'No worker thread captured'
print('threads',dict(threads),'events',len(events),'windows',len(windows))
print('top worker hotspots')
for x in [x for x in ranked if x['thread']=='worker'][:16]:print(x)
print('top main hotspots')
for x in [x for x in ranked if x['thread']=='main'][:8]:print(x)
