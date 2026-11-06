
from v26_common import *
import torch
assert (ROOT/'audit/v27_kvsharer_search_done.json').exists()
def quality(name,representation,n=500,fixed=False):
 a=argv(name,gpu=3,n=n,diagnostic=True,admission='reuse-prefetch',split='full_eval')
 a[a.index('--pcache-root')+1]=str(ROOT/'runtime_data/v27_pcache_gpu3')
 if fixed:
  end=a.index('--task')-1
  a[end:end]=['--logits',str(ROOT/f'runtime_data/{name}_logits'),'--selection-plan',str(ROOT/'runtime_data/v27_fixed2_I_logits')]
 cmd=[PY,'-u','scripts/run_baseline_quality_v27.py','--representation',str(ROOT/f'assets/{representation}/trec.pt'),'--',*a[3:]]
 run(name,cmd,3)
for g in [24,20,16]:
 quality(f'v27_kvsharer_g{g}_quality',f'v27_kvsharer_g{g}')
assert (ROOT/'audit/v27_full64_ready.json').exists()
quality('v27_clf_full64_bridge','v27_clf_full64',16,True)
def rows(n):return {x['uid']:x for x in map(json.loads,(ROOT/f'results/{n}/scored_records.jsonl').read_text().splitlines())}
base=rows('v27_fixed2_I');new=rows('v27_clf_full64_bridge')
assert set(base)==set(new)
diff=max(abs(base[k]['label_candidate_mean_logprobs'][l]-new[k]['label_candidate_mean_logprobs'][l]) for k in base for l in base[k]['label_candidate_mean_logprobs'])
pred=sum(base[k]['prediction']==new[k]['prediction'] for k in base)
report=dict(n=len(base),prediction_agreement=pred,max_label_score_difference=diff,quality_only=True,precision='float64',full32_bridge_failed_retained=True)
(ROOT/'audit/v27_clf_full64_bridge.json').write_text(json.dumps(report,indent=2))
assert pred==16 and diff<0.05,report
(ROOT/'audit/v27_clf_bridge_pass.json').write_text(json.dumps(report))
for tag in ['r256_384','r384_576']:
 quality('v27_clf_'+tag+'_quality','v27_clf_'+tag)
(ROOT/'audit/v27_baseline_quality_done.json').write_text(json.dumps({'n':500,'external_latency_valid':False}))
