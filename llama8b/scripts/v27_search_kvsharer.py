
"""KVSharer-Prefix: author distance/remap/threshold with fixed historical prefix.
Full query private KV stays native. Search uses only first 30 sorted FIT hashes.
No labels or evaluated quality used in acceptance. No SSD latency claim.
"""
from v26_common import *
import torch,torch.nn.functional as F
from transformers import AutoModelForCausalLM
from contiguous_fuxian.sharedkv.io import read_prefix
from contiguous_fuxian.sharedkv_v24.records import load
from contiguous_fuxian.sharedkv_v24.faithful import decoder
from contiguous_fuxian.sharedkv_v27.common import torch_new
from contiguous_fuxian.sharedkv_v27.baseline_quality import FORMAT
from contiguous_fuxian.sharedkv_v24.records import FIELDS
torch.set_num_threads(4);torch.set_grad_enabled(False)
fit=load(ROOT/'runtime_data/v24_teacher_fit','trec','fit');hashes=sorted(fit)[:30];head=fit[hashes[0]]
raw={i:tuple(read_prefix(ROOT/'runtime_data/prefix_store','trec',i,k)[0].to('cuda',torch.bfloat16) for k in ['key','value']) for i in range(28)}
edges=[]
for s in range(28):
 for t in range(s+1,28):
  # Author: mean over positions/heads of vector L2, K and V separately.
  distance=sum(float(torch.linalg.vector_norm(raw[s][j].float()-raw[t][j].float(),dim=-1).mean()) for j in [0,1])/2
  edges.append((distance,s,t))
edges.sort(key=lambda x:(-x[0],x[2],x[1]))
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
captured=[]
hook=model.model.norm.register_forward_hook(lambda m,a,o:captured.append(o.detach()))
identity=list(range(28))
def evaluate(h,sources):
 row=fit[h];banks={}
 for i,s in enumerate(sources):
  idx=torch.tensor(row['selection_by_layer'][i],device='cuda')
  banks[i]=tuple(x.index_select(0,idx).unsqueeze(0) for x in raw[s])
 captured.clear()
 logits,_=decoder(model,torch.tensor([row['query_token_ids']],device='cuda'),row['prefix_tokens'],None,identity,banks,{},False)
 hidden=captured[-1].float().flatten().clone()
 lp=logits[0,-1].float().log_softmax(-1)
 score_error=max(abs(float(lp[row['label_ids'][label][0]])-row['teacher_token_logprobs'][label][0]) for label in row['label_order'])
 return hidden,score_error
baseline={};parity=[]
for h in hashes:
 hidden,err=evaluate(h,identity);baseline[h]=hidden;parity.append(err)
print('identity replay max first-token logprob error',max(parity),flush=True)
# Faithful replay is expected bitwise at first token; fail on material mismatch.
assert max(parity)<0.02,parity
sources=identity.copy();history=[];accepted=set();snapshots={}
for distance,s,t in edges:
 if t in accepted:continue
 candidate=sources.copy();candidate[t]=s
 for i in range(28):
  if candidate[i]!=i:candidate[i]=candidate[candidate[i]]
 similarities=[]
 for h in hashes:
  hidden,_=evaluate(h,candidate)
  similarities.append(float(F.cosine_similarity(hidden[None],baseline[h][None])))
 mean=sum(similarities)/len(similarities);ok=mean>0.5
 history.append(dict(source=s,consumer=t,distance=distance,cosine=mean,min_cosine=min(similarities),accepted=ok))
 print('TRY',s,t,'cos',mean,'accept',ok,flush=True)
 if ok:
  sources=candidate;accepted.add(t)
  g=len(set(sources))
  if g in [24,20,16]:
   plan=dict(sources=sources.copy(),design_query_hashes=hashes,threshold=.5,achieved_sources=g,history=history.copy(),variant='KVSharer-Prefix author-distance-remap; fixed historical prefix; private query native; no truncation')
   snapshots[g]=plan
   (ROOT/f'configs/v27_kvsharer_g{g}.json').write_text(json.dumps(plan,indent=2))
  if g<=16:break
hook.remove()
(ROOT/'audit/v27_kvsharer_search.json').write_text(json.dumps(dict(history=history,identity_logprob_max_error=max(parity),sources=sources,achieved_sources=len(set(sources)),fit_hashes=hashes),indent=2))
for g,plan in snapshots.items():
 out=dict(format=FORMAT,**{k:head[k] for k in FIELDS},calibration_query_hashes=sorted(fit),method='direct-map-prefix-quality',quality_only=True,latency_valid=False,hardware_bytes_valid=False,position_offset=0,rope_theta=1e6,sources=plan['sources'],native={str(s):[x.cpu().float() for x in raw[s]] for s in sorted(set(plan['sources']))},map_provenance=plan)
 torch_new(ROOT/f'assets/v27_kvsharer_g{g}/trec.pt',out)
(ROOT/'audit/v27_kvsharer_search_done.json').write_text(json.dumps({'achieved_sources':len(set(sources)),'snapshots':list(snapshots)}))
