"""v37: Full-KV (100 % budget, identity mode) test accuracy for Llama-3.1-8B, same frozen test sets as every other method
(paper: Full KV reference line in the accuracy figure). Quality only, no timing. Run from the Llama tree with V30_* env."""
import runpy
from pathlib import Path
c = runpy.run_path(str(Path(__file__).resolve().parent / 'v30_common.py'))
R, PY = c['R'], c['PY']
OUT = R / 'studies/v37/fullkv'
plan = {'1': ['sst2', 'rte'], '2': ['subj'], '3': ['trec']}
queues = {}
for g, tasks in plan.items():
    queues[g] = []
    for t in tasks:
        o = OUT / 'runs' / f'fullkv_test_{t}'
        argv = [PY, c['entry'](False)] + c['chain'](o, task=t, budget=100, period=8, samples=c['N_TEST'][t],
                                                     bundle=c['test_bundle'](t, 25), pcache=c['PCACHE'][g])
        queues[g].append(dict(name=f'fullkv_test_{t}', argv=argv))
c['run_queues'](OUT, queues)
print('V37_FULLKV_DONE', flush=True)
