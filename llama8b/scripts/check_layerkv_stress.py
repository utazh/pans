#!/usr/bin/env python3
from pathlib import Path
import json,torch,statistics
R=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
details=[]
for p in sorted((R/'results/original_long_logits').glob('*.pt')):
 q=R/'results/dense1_long_logits'/p.name
 a=torch.load(p,weights_only=False);b=torch.load(q,weights_only=False)
 x=a['logits'].float();y=b['logits'].float()
 details.append({'query_hash':p.stem,'query_tokens':a['query_tokens'],
                 'original_completion':a['completion'],'shared_completion':b['completion'],
                 'logits_kl':(x.softmax(-1)*(x.log_softmax(-1)-y.log_softmax(-1))).sum().item()})
extra=[json.loads(x) for x in (R/'results/dense1_long.extra.jsonl').read_text().splitlines()]
report={'samples':len(details),'details':details,
        'query_token_range':[min(x['query_tokens'] for x in details),max(x['query_tokens'] for x in details)],
        'mean_logits_kl':statistics.mean(x['logits_kl'] for x in details),
        'generation_exact_match':statistics.mean(x['original_completion']==x['shared_completion'] for x in details),
        'private_cache_bounded':all(max(x['private_tail_lengths'].values())<=x['query_tokens']+3 for x in extra),
        'scope':'Artificial neutral-padding stress cases, not a natural task-quality benchmark.'}
(R/'results/long_query_stress.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))
