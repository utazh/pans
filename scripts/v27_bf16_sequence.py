
from v26_common import *
import torch,copy
from contiguous_fuxian.sharedkv_v27.common import torch_new
from contiguous_fuxian.sharedkv_v27.baseline_math import factor_bytes
assert (ROOT/'audit/v27_online_done.json').exists()
full=torch.load(ROOT/'assets/v27_clf_full64/trec.pt',weights_only=True)
for rk,rv in [(256,384),(384,576)]:
 b={k:v for k,v in full.items() if k!='groups'};groups=[]
 for g in full['groups']:
  gg={'layers':g['layers']}
  for kind,rank in [('key',rk),('value',rv)]:
   f=g[kind];s=f['left'].norm(dim=0)[:rank]
   left=(f['left'][:,:rank]/s[None]).bfloat16().contiguous()
   rights=[(s[:,None]*v[:rank]).bfloat16().contiguous() for v in f['rights']]
   gg[kind]=dict(left=left,rights=rights,geometry=f['geometry'],rank=rank)
  groups.append(gg)
 b.update(groups=groups,factor_bytes=sum(factor_bytes(g[k]) for g in groups for k in ['key','value']),factor_layout='U and S*Vh, BF16, author-style scale placement',variant='xKV-CLF-Prefix/P8; exact SVD reference then BF16; no landmark selector or SSD factor I/O',diagnostic_precision='bf16 factors from full64 SVD reference')
 torch_new(ROOT/f'assets/v27_clf_bf16_{rk}_{rv}/trec.pt',b)
del full,b,groups
def quality(name,rep,n=500,fixed=False):
 a=argv(name,gpu=2,n=n,diagnostic=True,admission='reuse-prefetch',split='full_eval')
 a[a.index('--pcache-root')+1]=str(ROOT/'runtime_data/v27_pcache_gpu2')
 if fixed:
  end=a.index('--task')-1;a[end:end]=['--logits',str(ROOT/f'runtime_data/{name}_logits'),'--selection-plan',str(ROOT/'runtime_data/v27_fixed2_I_logits')]
 run(name,[PY,'-u','scripts/run_baseline_quality_v27.py','--representation',str(ROOT/f'assets/{rep}/trec.pt'),'--',*a[3:]],2)
quality('v27_clf_full64_bridge_gpu2','v27_clf_full64',16,True)
def rows(n):return {x['uid']:x for x in map(json.loads,(ROOT/f'results/{n}/scored_records.jsonl').read_text().splitlines())}
a=rows('v27_clf_full64_bridge_gpu2');b=rows('v27_fixed2_I')
diff=max(abs(a[k]['label_candidate_mean_logprobs'][l]-b[k]['label_candidate_mean_logprobs'][l]) for k in a for l in a[k]['label_candidate_mean_logprobs'])
report=dict(max_label_score_difference=diff,n=len(a))
(ROOT/'audit/v27_full64_gpu2_bridge.json').write_text(json.dumps(report));print('full64gpu2',report,flush=True)
assert diff<0.05,report
for rk,rv in [(256,384),(384,576)]:
 quality(f'v27_clf_bf16_{rk}_{rv}_quality',f'v27_clf_bf16_{rk}_{rv}')
(ROOT/'audit/v27_bf16_done.json').write_text(json.dumps({'passed':True}))
