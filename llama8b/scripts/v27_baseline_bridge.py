
from v26_common import *
import torch
# Full-rank bridge is numerical validation, no timing; replay the same clean IDs.
assert (ROOT/'audit/v27_compile_done.json').exists()
name='v27_clf_full_bridge'
a=argv(name,gpu=3,n=16,diagnostic=True,admission='reuse-prefetch',split='full_eval')
a[a.index('--pcache-root')+1]=str(ROOT/'runtime_data/v27_pcache_gpu3')
end=a.index('--task')-1
a[end:end]=['--logits',str(ROOT/f'runtime_data/{name}_logits'),'--selection-plan',str(ROOT/'runtime_data/v27_fixed2_I_logits')]
cmd=[PY,'-u','scripts/run_baseline_quality_v27.py','--representation',str(ROOT/'assets/v27_clf_full/trec.pt'),'--',*a[3:]]
run(name,cmd,3)
base={json.loads(l)['uid']:json.loads(l) for l in (ROOT/'results/v27_fixed2_I/scored_records.jsonl').read_text().splitlines()}
new={json.loads(l)['uid']:json.loads(l) for l in (ROOT/f'results/{name}/scored_records.jsonl').read_text().splitlines()}
assert set(base)==set(new) and len(base)==16
diff=max(abs(base[k]['label_candidate_mean_logprobs'][l]-new[k]['label_candidate_mean_logprobs'][l]) for k in base for l in base[k]['label_candidate_mean_logprobs'])
pred=sum(base[k]['prediction']==new[k]['prediction'] for k in base)
report=dict(n=16,prediction_agreement=pred,max_label_score_difference=diff,quality_only=True)
(ROOT/'audit/v27_clf_bridge.json').write_text(json.dumps(report,indent=2))
assert pred==16 and diff<0.05,report
(ROOT/'audit/v27_clf_bridge_pass.json').write_text(json.dumps(report))
print('BRIDGE PASS',report,flush=True)
run('v27_kvsharer_search',[PY,'-u','scripts/v27_search_kvsharer.py'],3)
