from v26_common import *
rows=[]
for pref,task,budget in [(p,t,b) for p in ['p0','p1'] for t in ['sst2','subj','trec','rte'] for b in ([5,10,25,50] if p=='p0' else [25])]:
 tag=f'v26_{pref}_{task}_b{budget}'+('_canonical' if task=='subj' else '')
 done=[ROOT/f'audit/{tag}_quality_{a}_process.json' for a in ['I','S']]
 if not all(p.exists() and json.loads(p.read_text())['returncode']==0 for p in done):continue
 scores=[]
 for arm in ['I','S']:
  data=[json.loads(x) for x in (ROOT/f'results/{tag}_quality_{arm}/scored_records.jsonl').read_text().splitlines()]
  scores.append(sum(x['correct'] for x in data));n=len(data)
 rows.append(dict(prefix=pref,task=task,budget=budget,n=n,identity=scores[0],shared=scores[1],delta_pp=round(100*(scores[1]-scores[0])/n,3)))
state=dict(status='incomplete live snapshot, not final report',time=time.time(),complete_main_quality_pairs=len(rows),rows=rows)
(ROOT/'audit/v26_live_quality_snapshot.json').write_text(json.dumps(state,indent=2));print(json.dumps(state))
