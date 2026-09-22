
from pathlib import Path
import json,torch
r=Path(__file__).resolve().parents[1];report={}
for arm in ['identity','shared']:
 a=r/f'runtime_data/v25_bridge_v24_{arm}_logits';b=r/f'runtime_data/v25_bridge_v25_{arm}_logits'
 aa={p.name:p for p in a.glob('*.pt')};bb={p.name:p for p in b.glob('*.pt')}
 assert len(aa)==8 and set(aa)==set(bb)
 errors=[];scores=[];plans=[]
 for k in aa:
  x=torch.load(aa[k],map_location='cpu',weights_only=True);y=torch.load(bb[k],map_location='cpu',weights_only=True)
  errors.append(float((x['logits'].float()-y['logits'].float()).abs().max()))
 xa=[json.loads(v) for v in (r/f'results/v25_bridge_v24_{arm}/scored_records.jsonl').read_text().splitlines()]
 ya=[json.loads(v) for v in (r/f'results/v25_bridge_v25_{arm}/scored_records.jsonl').read_text().splitlines()]
 assert [v['uid'] for v in xa]==[v['uid'] for v in ya]
 for x,y in zip(xa,ya):
  assert x['layer_token_selection_sha256']==y['layer_token_selection_sha256']
  for label,values in x['label_candidate_token_logprobs'].items():
   scores.extend(abs(u-v) for u,v in zip(values,y['label_candidate_token_logprobs'][label]))
 assert max(errors)==0 and max(scores)==0,(arm,errors,max(scores))
 report[arm]={'queries':8,'max_first_token_full_vocab_logit_error':max(errors),'max_continuation_token_score_error':max(scores),'selection_hash_exact':True}
p=r/'audit/v25_smoke_r4_online_parity.jsonl';z=[json.loads(x) for x in p.read_text().splitlines()]
assert len(z)==8 and all(x['max_token_error']<=.10 and x['max_mean_score_error']<=.02 for x in z)
report['r4_online_replay']={'queries':8,'max_token_error':max(x['max_token_error'] for x in z),'max_mean_score_error':max(x['max_mean_score_error'] for x in z)}
with (r/'audit/v25_bridge_validation.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
