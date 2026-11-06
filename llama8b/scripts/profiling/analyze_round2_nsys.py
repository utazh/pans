import sqlite3,json,collections,bisect,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('profile_dir',type=Path);a=p.parse_args();base=a.profile_dir.resolve()
c=sqlite3.connect(base/'nsys.sqlite');c.row_factory=sqlite3.Row
strings=dict(c.execute('select id,value from StringIds'))
nv=[dict(x) for x in c.execute('select * from NVTX_EVENTS')]
for e in nv:e['name']=e['text'] or strings.get(e['textId'],'')
ranges=[e for e in nv if e['end'] is not None and e['end']>=e['start']]
windows=sorted([e for e in ranges if e['name']=='TTFT/logits_ready'],key=lambda e:e['start'])
assert len(windows)==4, f'Expected four TTFT windows, found {len(windows)}'
print('TTFT windows',[(e['start'],e['end'],(e['end']-e['start'])/1e6) for e in windows])
apis={e['correlationId']:dict(e) for e in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')}
bytid=collections.defaultdict(list)
for e in ranges:
 if not e['name'].startswith('request/') and not e['name'].startswith('TTFT/'):bytid[e['globalTid']].append(e)
for es in bytid.values():es.sort(key=lambda e:e['start'])
starts={tid:[e['start'] for e in es] for tid,es in bytid.items()}
def ancestors(api):
 if api is None:return []
 tid=api['globalTid'];ts=api['start'];es=bytid.get(tid,[])
 idx=bisect.bisect_right(starts.get(tid,[]),ts)
 return [e['name'] for e in es[max(0,idx-100):idx] if e['end']>=ts]
def category(names,default='other'):
 if any(n.startswith('decoder/') for n in names):return 'decoder'
 if 'qwen_online_prefix_head_scores' in names:return 'selector_score'
 if 'index/H2D_and_dequantize' in names:return 'selector_index'
 if 'configure_online_layer_selection' in names:return 'selector_control'
 if '_cache_layout' in names:return 'cache_layout'
 if '_layer_causal_mask' in names:return 'attention_mask'
 return default
def union(intervals):
 out=[]
 for a,b in sorted(intervals):
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
  else:out.append((a,b))
 return out
def length(intervals):return sum(b-a for a,b in union(intervals))/1e6
def intersect(a,b):
 a=union(a);b=union(b);i=j=0;out=[]
 while i<len(a) and j<len(b):
  left=max(a[i][0],b[j][0]);right=min(a[i][1],b[j][1])
  if right>left:out.append((left,right))
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return out
gpu=[]
for table,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMCPY','memcpy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
 for row in c.execute('select * from '+table):
  e=dict(row);e['kind']=kind;e['category']=category(ancestors(apis.get(e.get('correlationId'))),'payload_or_other')
  if kind=='kernel':e['name']=strings.get(e['shortName'],str(e['shortName']))
  gpu.append(e)
labels={
'configure_online_layer_selection':'selector_control_and_sync',
'qwen_online_prefix_head_scores':'selector_score_enqueue',
'prepare_impress_block_scores':'score_preparation',
'select_promixed_gqa_blocks':'selection_decision',
'load_selector_keys':'selector_ready_wait',
'_resolve_entry':'payload_wait_and_gather',
'resolve':'payload_resolve_other',
'_resolve_impress':'payload_resolve_other',
'_update_loaded_layer_score':'cache_score_maintenance',
'_cache_layout':'payload_layout_cast',
'_layer_causal_mask':'attention_mask',
'resolve_deferred_impress_compute':'event_harvest',
}
results=[]
for request_id,w in enumerate(windows,1):
 a,b=w['start'],w['end'];tid=w['globalTid']
 stages=[]
 for e in bytid[tid]:
  if e['end']<=a or e['start']>=b:continue
  name=e['name']
  label=labels.get(name)
  if name.startswith('decoder/'):label='decoder_enqueue'
  elif name.startswith('schedule') or name=='prefetch_selector_keys':label='prefetch_scheduling'
  if label:stages.append((max(a,e['start']),min(b,e['end']),label,e['end']-e['start']))
 points=sorted({a,b}|{v for e in stages for v in e[:2]})
 buckets=collections.defaultdict(float);timeline=[]
 for l,h in zip(points,points[1:]):
  active=[e for e in stages if e[0]<=l and e[1]>=h]
  label=min(active,key=lambda e:e[3])[2] if active else 'other_control_and_sync'
  buckets[label]+=(h-l)/1e6
  if timeline and timeline[-1][2]==label:timeline[-1][1]=h
  else:timeline.append([l,h,label])
 selected=[e for e in gpu if e['start']<b and e['end']>a]
 kernels=[(max(a,e['start']),min(b,e['end'])) for e in selected if e['kind']=='kernel']
 copies=[(max(a,e['start']),min(b,e['end'])) for e in selected if e['kind']=='memcpy']
 busy=[(max(a,e['start']),min(b,e['end'])) for e in selected]
 catspans=collections.defaultdict(list);streams=collections.defaultdict(list);copy_info=collections.defaultdict(lambda:{'calls':0,'bytes':0,'sum_ms':0.})
 kernel_names=collections.defaultdict(lambda:[0,0.])
 for e in selected:
  span=(max(a,e['start']),min(b,e['end']))
  catspans[e['category']].append(span);streams[str(e['streamId'])].append(span)
  if e['kind']=='memcpy':
   v=copy_info[str(e['copyKind'])];v['calls']+=1;v['bytes']+=e['bytes'];v['sum_ms']+=(span[1]-span[0])/1e6
  if e['kind']=='kernel':
   v=kernel_names[e['name']];v[0]+=1;v[1]+=(span[1]-span[0])/1e6
 api_sync=collections.defaultdict(lambda:[0,0.])
 for e in apis.values():
  if e['globalTid']==tid and e['start']<b and e['end']>a:
   name=strings[e['nameId']]
   if 'Synchronize' in name:
    v=api_sync[name];v[0]+=1;v[1]+=(min(b,e['end'])-max(a,e['start']))/1e6
 overlaps={}
 for left,right in [('decoder','selector_index'),('decoder','payload_or_other'),('selector_score','payload_or_other'),('selector_control','selector_index')]:
  overlaps[left+'__'+right]=length(intersect(catspans[left],catspans[right]))
 item=dict(request=request_id,wall_ms=(b-a)/1e6,cpu_partition_ms=dict(buckets),
  gpu_busy_union_ms=length(busy),gpu_no_activity_ms=(b-a)/1e6-length(busy),
  kernel_union_ms=length(kernels),copy_union_ms=length(copies),kernel_copy_overlap_ms=length(intersect(kernels,copies)),
  gpu_category_union_ms={k:length(v) for k,v in catspans.items()},stream_busy_union_ms={k:length(v) for k,v in streams.items()},
  copy_kinds=dict(copy_info),gpu_overlaps_ms=overlaps,sync_api_calls_and_ms=dict(api_sync),
  kernel_count=sum(e['kind']=='kernel' for e in selected),
  top_kernels=sorted([{'name':k,'calls':v[0],'sum_ms':v[1]} for k,v in kernel_names.items()],key=lambda e:-e['sum_ms'])[:15])
 if abs(sum(buckets.values())-item['wall_ms'])>1e-6:raise ValueError('CPU partition failed')
 item['main_nvtx_counts']=dict(collections.Counter(e['name'] for e in bytid[tid] if e['start']>=a and e['end']<=b))
 assert item['main_nvtx_counts'].get('qwen_online_prefix_head_scores',0)==4,'Expected 4 P8 selectors'
 assert sum(v for k,v in item['main_nvtx_counts'].items() if k.startswith('decoder/layer_'))==28,'Expected 28 decoder layers'
 item['gpu_overlap_cpu_phase_ms']={label:length(intersect(busy,[(l,h) for l,h,name in timeline if name==label])) for label in buckets}
 all_api=collections.defaultdict(lambda:[0,0.])
 for api in apis.values():
  if api['start']<b and api['end']>a:
   key=('main' if api['globalTid']==tid else 'worker',strings[api['nameId']])
   v=all_api[key];v[0]+=1;v[1]+=(min(b,api['end'])-max(a,api['start']))/1e6
 item['cuda_api_hotspots']=[dict(thread=k[0],name=k[1],calls=v[0],cpu_elapsed_ms=v[1]) for k,v in sorted(all_api.items(),key=lambda x:-x[1][1])]
 results.append(item)
 if request_id==1:
  compact={'window_ns':[a,b],'cpu_timeline':timeline,'gpu_events':[{k:e[k] for k in ['start','end','kind','category','streamId']} for e in selected],
   'worker_nvtx':[{'start':e['start'],'end':e['end'],'name':e['name'],'tid':e['globalTid']} for e in ranges if e['globalTid']!=tid and e['start']<b and e['end']>a]}
  (base/'ttft_timeline_data.json').write_text(json.dumps(compact))
(base/'nsys_ttft_breakdown.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
