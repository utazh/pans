import json
from pathlib import Path
R=Path('/home/panzihang/src/prism_layerkv')
arms={'FULLKV':'v29:FULLKV','NS25':'v29:I','G24':'v29:syn_G24','G20':'v29:syn_G20','G16':'v29:syn_G16','NSP4_25':'v30:NSP4_25'}
for t in ('21p4286','17p8571','14p2857'): arms['NS_'+t]='v30:NS_'+t
for m in ('CAKE','LAVA'):
    for t in ('25','21p4286','17p8571','14p2857'): arms[f'{m}_{t}']=f'v30:{m}_{t}'
def d(src,name,s,slot): return (R/f'studies/v29/quality/{s}/eval/{name}_{slot}') if src=='v29' else (R/f'studies/v30/quality_qa/runs/{s}_{slot}_{name}')
done=[]
for s in ('f0','f1','f2'):
    for slot in ('sst2','subj','trec','rte'):
        if all((d(*v.split(':'),s,slot)/'summary.json').exists() for v in arms.values()): done.append((s,slot))
print('complete articles',len(done),done)
for a,v in arms.items():
    c=n=0;mb=0
    for s,slot in done:
        p=d(*v.split(':'),s,slot); rows=[json.loads(l) for l in (p/'scored_records.jsonl').read_text().splitlines()]
        c+=sum(r['correct'] for r in rows); n+=len(rows)
        t=list(json.loads((p/'summary.json').read_text())['tasks'].values())[0]; mb+=(t.get('mean_physical_prefetch_kv_bytes') or 0)/1e6
    if n: print(f'{a:12s} acc {100*c/n:6.2f}  n {n}  MB {mb/len(done):7.1f}')
