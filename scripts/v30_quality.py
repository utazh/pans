"""v30 quality campaign on Qwen2.5-7B (3090). Groups (run any subset, resumable, never overwrites):
  ns_p4_dev    NS (identity) P4, dev500, 4 tasks x budgets 5/10/25/50        (P8 dev exists: studies/v29/a0/runs/devI_*)
  ns_p4_test   NS P4, frozen test set, same 16 cells                          (P8 test exists: studies/v29/test_quality)
  ns_eq        NS P8 and P4 at byte-aligned budgets 14.2857/17.8571/21.4286 (= 25% x 16/20/24 of 28 banks), test
  p4_dev       RAPS P4 candidates G16/G20/G24 at 25%, dev500 (selection input)
  p4_test      RAPS P4 candidates G16/G20/G24 at 25%, test
  cake_static / lava_static   prefix-window CAKE / LAVa chunk selection, P8 engine, budgets 10/14.29/17.86/21.43/25/50, test
  cake_budget / lava_budget   CAKE / LAVa layer budgets only (online selector picks chunks), 14.29 and 25, test
  cake_oracle / lava_oracle   request-window CAKE / LAVa with full prefix keys, 25%, test (quality reference only)
Usage: v30_quality.py <group[,group...]> <gpu[,gpu...]>"""
from pathlib import Path
import json, runpy, sys
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY = C['R'], C['PY']; TASKS, N = C['TASKS'], C['N_TEST']
OUT = R / 'studies/v30/quality'
EQ = [100 * 16 / 28 / 4, 100 * 20 / 28 / 4, 100 * 24 / 28 / 4]
STATIC_B = [10, *EQ, 25, 50]
tag = lambda b: f'{b:.4f}'.rstrip('0').rstrip('.').replace('.', 'p')
P4R = {t: R / f'studies/v30/p4_readers/{t}/readers' for t in ('sst2', 'trec', 'rte')}
P4R['subj'] = R / ('studies/v30/p4_readers/subj/readers' if __import__('os').environ.get('V30_ROOT') else 'studies/v29/subj_p4/readers')

def test_bundle(task):
    b = C['test_bundle'](task)
    ref = C['sha'](b / f'{task}.jsonl')
    for bb in (5, 10, 50):   # the eval split must be identical across budgets of a task
        assert C['sha'](C['test_bundle'](task, bb) / f'{task}.jsonl') == ref, (task, bb)
    return b

def jobs(group):
    out = []
    def add(name, task, budget, period, split, adapters=None, ext=(), ext_needed=False):
        bundle = C['DEV'] if split == 'dev' else test_bundle(task)
        samples = 500 if split == 'dev' else N[task]
        out.append(dict(name=name, task=task, budget=budget, period=period, split=split, adapters=adapters, ext=list(ext),
                        ext_needed=ext_needed or bool(ext) or float(budget) != int(budget), bundle=bundle, samples=samples))
    for t in TASKS:
        if group in ('ns_p4_dev', 'ns_p4_test'):
            for b in (5, 10, 25, 50): add(f'{group}_{t}_k{b}', t, b, 4, group[-4:].strip('_'))
        elif group == 'ns_p8_test':   # NS P8 at 25 % (the original main configuration; 7B has it in studies/v29/test_quality)
            add(f'ns_p8_test_{t}_k25', t, 25, 8, 'test')
        elif group == 'ns_eq':
            for p in (8, 4):
                for b in EQ: add(f'ns_eq_{t}_k{tag(b)}_p{p}', t, b, p, 'test')
        elif group in ('p4_dev', 'p4_test'):
            cands = sorted(d.name for d in P4R[t].iterdir() if (d / f'{t}.pt').exists())   # G16/G20/G24, SST-2: G21/G24
            assert cands, f'no P4 readers for {t}'
            for g in cands: add(f'{group}_{g}_{t}_k25', t, 25, 4, group[3:], adapters=P4R[t] / g)
        elif group.endswith('_static'):
            m = group.split('_')[0]
            for b in STATIC_B:
                add(f'{m}_static_{t}_k{tag(b)}', t, b, 8, 'test', ext=['--static-selection', str(R / f'configs/v30_sel/{m}_prefix_{t}_k{tag(b)}.json')])
        elif group.endswith('_budget'):
            m = group.split('_')[0]
            for b in (EQ[0], 25):
                add(f'{m}_budget_{t}_k{tag(b)}', t, b, 8, 'test', ext=['--layer-budget-profile', str(R / f'configs/v30_sel/{m}budget_{t}_k{tag(b)}.json')])
        elif group.endswith('_oracle'):
            m = group.split('_')[0]
            add(f'{m}_oracle_{t}_k25', t, 25, 8, 'test', ext=['--oracle-selection-dir', str(R / f'runtime_data/v30_oracle/{m}_{t}_k25_test')])
        else: raise SystemExit(f'unknown group {group}')
    return out

groups, gpus = sys.argv[1].split(','), sys.argv[2].split(',')
js = [j for g in groups for j in jobs(g)]
queues = {g: [] for g in gpus}
for i, j in enumerate(js):
    g = gpus[i % len(gpus)]; o = OUT / 'runs' / j['name']
    argv = [PY, C['entry'](j['ext_needed']), *C['chain'](o, task=j['task'], budget=j['budget'], period=j['period'], samples=j['samples'],
                                                         bundle=j['bundle'], pcache=C['PCACHE'][g], adapters=j['adapters'], ext=j['ext'])]
    queues[g].append(dict(name=j['name'], argv=argv))
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / 'plan.jsonl').open('a') as f:
    for g, q in queues.items():
        for j in q: f.write(json.dumps(dict(gpu=g, **j), default=str) + '\n')
print({g: len(q) for g, q in queues.items()}, flush=True)
C['run_queues'](OUT, queues)
print('V30_QUALITY_GROUPS_DONE', groups, flush=True)
