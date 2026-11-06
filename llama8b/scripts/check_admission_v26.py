
from v26_common import *
import torch
report={}
for cold in ['warm','cold']:
 for kind in ['I','S']:
  names=[f'v26_audit3_{cold}_{kind}_{mode}' for mode in ['original','reuse-prefetch']]
  folders=[ROOT/'audit'/x for x in names]
  paths=[sorted(x.glob('request*.pt')) for x in folders]
  assert len(paths[0])==len(paths[1])==10
  rows=[]
  for a,b in zip(*paths):
   x=torch.load(a,weights_only=True);y=torch.load(b,weights_only=True)
   assert torch.equal(x['first_logits'],y['first_logits']),('logits',a,b)
   for k in ['selection_by_layer','label_token_logprobs','query_token_ids','cache_state_sha256','resident_values_sha256','heap_order','gpu_mb','cpu_mb','decoder_calls','all28_private']:
    assert x[k]==y[k],(cold,kind,k)
   rows.append(dict(max_logit_error=float((x['first_logits'].float()-y['first_logits'].float()).abs().max()),promotions_verified=y['promotions_verified'],original_peak=x['peak_allocated'],reuse_peak=y['peak_allocated'],reuse_registry_backing_peak=y['peak_registry_backing_bytes']))
  assert sum(x['promotions_verified'] for x in rows)>0
  report[cold+'-'+kind]=rows
  if cold=='cold':
   for name in names:
    lines=[json.loads(x) for x in (ROOT/f'results/{name}.cold.jsonl').read_text().splitlines()]
    assert len(lines)==8 and all(x['status']=='completed' and x['application_payload_empty'] and x['page_after']['resident']==0 and x['gpu_mib_after']==0 and x['cpu_mib_after']==0 for x in lines)
    report[name+'_cold_log']=lines
with (ROOT/'audit/v26_admission_gpu_parity.json').open('x') as f:json.dump(dict(passed=True,comparisons=report,scope='8 measured + 2 warmup per arm, warm and SSD-cold/preserve; exact whole-vocabulary last-query logits, every candidate token, IDs, full cache residency/scores/heap order/value hashes; verified independent block storage, all 28 private KV'),f,indent=2)
print('ADMISSION PARITY PASSED')
