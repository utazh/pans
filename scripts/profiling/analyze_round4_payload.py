import json,sqlite3,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('profile_dir',type=Path);base=p.parse_args().profile_dir
c=sqlite3.connect(base/'nsys.sqlite');c.row_factory=sqlite3.Row
strings=dict(c.execute('select id,value from StringIds'))
events=[dict(x) for x in c.execute('select * from NVTX_EVENTS')]
for e in events:e['name']=e['text'] or strings.get(e['textId'],'')
events=[e for e in events if e['end'] is not None and e['end']>=e['start']]
windows=sorted([e for e in events if e['name']=='TTFT/logits_ready'],key=lambda e:e['start'])
assert len(windows)==4
def union(spans):
 out=[]
 for a,b in sorted(spans):
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
  else:out.append((a,b))
 return out
def length(spans):return sum(b-a for a,b in union(spans))/1e6
def intersection(a,b):
 a=union(a);b=union(b);i=j=0;out=[]
 while i<len(a) and j<len(b):
  l=max(a[i][0],b[j][0]);h=min(a[i][1],b[j][1])
  if h>l:out.append((l,h))
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return out
gpu_spans={}
for kind,table in [('kernel','CUPTI_ACTIVITY_KIND_KERNEL'),('copy','CUPTI_ACTIVITY_KIND_MEMCPY'),('memset','CUPTI_ACTIVITY_KIND_MEMSET')]:
 gpu_spans[kind]=[(e['start'],e['end']) for e in c.execute('select start,end from '+table)]
rows=[]
for n,w in enumerate(windows,1):
 a,b=w['start'],w['end']
 spans={}
 counts={}
 for label in ['payload/prepare_and_pack','payload/host_gather','metadata/source_counts']:
  selected=[e for e in events if e['name']==label and e['start']<b and e['end']>a]
  spans[label]=[(max(a,e['start']),min(b,e['end'])) for e in selected]
  counts[label]=len(selected)
 pack,gather=spans['payload/prepare_and_pack'],spans['payload/host_gather']
 waits=[(max(a,e['start']),min(b,e['end'])) for e in events if e['name']=='_resolve_entry' and e['globalTid']==w['globalTid'] and e['start']<b and e['end']>a]
 row=dict(request=n,wall_ms=(b-a)/1e6,prepare_pack_union_ms=length(pack),host_gather_union_ms=length(gather),
  prepare_pack_excluding_host_gather_ms=length(pack)-length(intersection(pack,gather)),
  source_accounting_union_ms=length(spans['metadata/source_counts']),
  gather_overlap_main_resolve_ms=length(intersection(gather,waits)),calls=counts)
 busy=union([span for spans_ in gpu_spans.values() for span in spans_])
 row['host_gather_overlap_gpu_kernel_ms']=length(intersection(gather,gpu_spans['kernel']))
 row['host_gather_overlap_gpu_busy_ms']=length(intersection(gather,busy))
 row['prepare_pack_overlap_gpu_busy_ms']=length(intersection(pack,busy))
 assert 0<=row['host_gather_overlap_gpu_kernel_ms']<=row['host_gather_overlap_gpu_busy_ms']+1e-6
 assert row['host_gather_overlap_gpu_busy_ms']<=row['host_gather_union_ms']+1e-6
 assert row['prepare_pack_excluding_host_gather_ms']>=-1e-6
 rows.append(row)
(base/'payload_batch_breakdown.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
