
from v26_common import *
from contiguous_fuxian.sharedkv_v26.analysis import records,paired_time_blocks
p=ROOT/'results/v26_primary_study/study.json';spec=json.loads(p.read_text());idx={}
for j in spec['jobs']:idx[j['condition'],j['block'],j['arm']]=(j,records(Path(j['output'])/'scored_records.jsonl'))
out={}
for condition in ['preserve','warm']:
 for title,left,right in [('shared/original','I-original','S-original'),('shared/reuse','I-reuse','S-reuse'),('admission/I','I-original','I-reuse'),('admission/S','S-original','S-reuse')]:
  pairs=[dict(block=b,baseline=idx[condition,b,left][1],student=idx[condition,b,right][1]) for b in range(8)]
  result=paired_time_blocks(pairs)
  strata={}
  for b in range(8):
   first=idx[condition,b,left][0]['position_in_block']<idx[condition,b,right][0]['position_in_block']
   strata.setdefault('left_first' if first else 'right_first',[]).append(result['per_block'][b])
  result['order_strata']=strata;out[condition+'/'+title]=result
 for kind in ['I','S']:
  for b in range(8):
   x,y=idx[condition,b,kind+'-original'][1],idx[condition,b,kind+'-reuse'][1]
   assert x.keys()==y.keys()
   for k in x:
    for field in ['prediction','correct','label_candidate_token_logprobs','layer_token_selection_sha256']:
     assert x[k][field]==y[k][field],(condition,kind,b,k,field)
 for arm in ['I-original','S-original','I-reuse','S-reuse']:
  if condition!='preserve':continue
  for b in range(8):
   job=idx[condition,b,arm][0]
   logs=[json.loads(x) for x in Path(job['output']+'.cold.jsonl').read_text().splitlines()]
   assert len(logs)==16 and all(x['status']=='completed' and x['application_payload_empty'] and x['page_after']['resident']==0 for x in logs)
with (ROOT/'results/v26_primary_complete_interim.json').open('x') as f:json.dump(out,f,indent=2)
print('PRIMARY ANALYZED; admission labels/IDs exact; 512 cold certificates')
