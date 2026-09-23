
from v26_common import *
import torch,statistics
from contiguous_fuxian.sharedkv_v27.baseline_math import factor_bytes
plans=[torch.load(p,weights_only=True)['selected_by_layer'] for p in (ROOT/'runtime_data/v27_fixed2_I_logits').glob('*.pt')]
result={'scope':'Representation accounting on 16 fixed identity plans; projected selective I/O is not implemented external I/O; no external latency claims','reference_dtype_bytes':2,'methods':{}}
result['identity_selected_bytes']=statistics.mean(sum(len(x) for x in plan)*512*2*2 for plan in plans)
for g in [24,20,16]:
 bundle=torch.load(ROOT/f'assets/v27_kvsharer_g{g}/trec.pt',weights_only=True)
 src=bundle['sources'];amount=[]
 for plan in plans:
  unions={s:set() for s in set(src)}
  for lid,s in enumerate(src):unions[s].update(plan[lid])
  amount.append(sum(map(len,unions.values()))*512*2*2)
 result['methods'][f'KVSharer-Prefix_G{g}']=dict(sources=src,complete_sources=g,projected_union_selected_bf16_bytes=statistics.mean(amount),quality_bridge_resident_fp32_native_bytes=sum(x.numel()*x.element_size() for xs in bundle['native'].values() for x in xs),actual_speed_available=False)
for tag in ['r256_384','r384_576']:
 b=torch.load(ROOT/f'assets/v27_clf_{tag}/trec.pt',weights_only=True)
 amount=[]
 for plan in plans:
  n=0
  for group in b['groups']:
   ids=set().union(*(set(plan[i]) for i in group['layers']))
   for k in ['key','value']:
    f=group[k];n+=len(ids)*f['rank']*f['left'].element_size()
    n+=sum(x.numel()*x.element_size() for x in f['rights'])
  amount.append(n)
 result['methods']['xKV-CLF_'+tag]=dict(full_fp32_factor_bytes=b['factor_bytes'],projected_selected_factor_bytes=statistics.mean(amount),dtype='float32',actual_speed_available=False,selected_reconstruction_workspace_per_layer_bytes=1244*512*2*2)
for arm in ['native','r0']:
 base=[json.loads(l) for l in (ROOT/f'results/v27_fixed2_{arm}.extra.jsonl').read_text().splitlines()]
 side=[json.loads(l) for l in (ROOT/f'results/v27_fixed2_{arm}.extra.jsonl.v25.jsonl').read_text().splitlines()]
 result['methods'][arm]={'first_base':base[0],'first_native_audit':side[0]}
(ROOT/'results/v27_representation_accounting.json').write_text(json.dumps(result,indent=2))
print('ACCOUNTING',json.dumps({k:{kk:vv for kk,vv in v.items() if 'bytes' in kk or kk=='complete_sources'} for k,v in result['methods'].items()}))
