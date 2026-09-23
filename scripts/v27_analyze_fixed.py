
from v26_common import *
import torch
arms=['I','native','proxy','r0'];data={}
for arm in arms:
 data[arm]={p.stem:torch.load(p,weights_only=True) for p in (ROOT/f'runtime_data/v27_fixed2_{arm}_logits').glob('*.pt')}
assert all(set(x)==set(data['I']) and len(x)==16 for x in data.values())
for arm in arms[1:]:
 assert all(data[arm][h]['selected_by_layer']==data['I'][h]['selected_by_layer'] for h in data['I'])
for arm in arms:
 assert all(torch.isfinite(x['logits']).all() for x in data[arm].values())
out={}
for a,b in [('I','native'),('I','proxy'),('I','r0'),('proxy','r0')]:
 kls=[];delta=[]
 for h in data[a]:
  x=data[a][h]['logits'].float().flatten();y=data[b][h]['logits'].float().flatten()
  p=x.log_softmax(-1);q=y.log_softmax(-1)
  kls.append(float((p.exp()*(p-q)).sum()));delta.append(float((x-y).abs().max()))
 out[a+'_vs_'+b]=dict(mean_first_token_kl=sum(kls)/len(kls),max_logit_difference=max(delta))
for arm in arms:
 rows=[json.loads(x) for x in (ROOT/f'results/v27_fixed2_{arm}/scored_records.jsonl').read_text().splitlines()]
 out[arm]=dict(n=len(rows),correct=sum(x['correct'] for x in rows))
stats=[json.loads(l) for l in (ROOT/'results/v27_fixed2_r0.extra.jsonl').read_text().splitlines()]
for x in stats:
 assert x['physical_resolve_calls']==16
 for i,s in enumerate(x['sources']):assert x['prefix_storage_ptrs_by_layer'][i]==x['prefix_storage_ptrs_by_layer'][s]
 assert len(x['private_tail_lengths'])==28
out['fixed_support_verified']=True;out['quality_certified']=False
(ROOT/'audit/v27_fixed_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
