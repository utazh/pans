"""v30 SSD-cold timing on the 3090 (v29 timing rule rev 3 executor: GPU 2, cores 28-31,60-63, pre-block gate, telemetry,
flagged/failed block rerun once). Cell = <task>_b25, 6 blocks x 16 queries, rotated arm orders, same SSD pcache copy for all
arms (runtime_data/v29_pcache_gpu2_ssd). Arms:
  NS25 / NS21 / NS18 / NS14   identity P8 at 25 and byte-aligned 21.4286 / 17.8571 / 14.2857 %
  G24 / G20 / G16             RAPS P8 candidates (frozen v29 readers)
  CAKE / LAVA                 prefix-window static selections at 25 %
  NS25P4 / SELP4              identity P4 at 25 %, dev-selected RAPS P4 candidate (from results/v30/p4_selection.json)
Usage: v30_timing.py [--dry-run] [--tasks sst2,subj,trec,rte]"""
from pathlib import Path
import json, runpy, sys
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY = C['R'], C['PY']
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py'))
C16 = R / 'runtime_data/v29_pcache_gpu2_ssd'
EQ = {'NS21': 100 * 24 / 28 / 4, 'NS18': 100 * 20 / 28 / 4, 'NS14': 100 * 16 / 28 / 4}
man = json.loads((R / 'studies/v29/readers/manifest.json').read_text())['cells']
p4sel = json.loads((R / 'results/v30/p4_selection.json').read_text())['cells']
P4R = {t: R / f'studies/v30/p4_readers/{t}/readers' for t in ('sst2', 'trec', 'rte')}; P4R['subj'] = R / 'studies/v29/subj_p4/readers'
tasks = sys.argv[sys.argv.index('--tasks') + 1].split(',') if '--tasks' in sys.argv else C['TASKS']
ARMS = ['NS25', 'NS21', 'NS18', 'NS14', 'G24', 'G20', 'G16', 'CAKE', 'LAVA', 'NS25P4', 'SELP4']

def arm_spec(task, arm):
    if arm == 'SELP4':
        s = p4sel[f'{task}_b25']['selected']
        return (dict(budget=25, period=4) if s == 'I' else dict(budget=25, period=4, adapters=P4R[task] / s)), False
    if arm == 'NS25': return dict(budget=25, period=8), False
    if arm == 'NS25P4': return dict(budget=25, period=4), False
    if arm in EQ: return dict(budget=EQ[arm], period=8), True
    if arm in ('G24', 'G20', 'G16'): return dict(budget=25, period=8, adapters=man[f'{task}_b25'][arm]['adapter_dir']), False
    m = arm.lower()
    return dict(budget=25, period=8, ext=['--static-selection', str(R / f'configs/v30_sel/{m}_prefix_{task}_k25.json')]), True

def argv(out, cell, arm, cond, block, name):
    task = cell.split('_')[0]; spec, ext = arm_spec(task, arm); o = out / 'runs' / name
    return [PY, C['entry'](ext), *C['chain'](o, task=task, samples=16, warmup=4, cold=cond, pcache=C16,
                                             bundle=R / f'studies/v28_frozen/bundles/{cell}/b{block:02d}', **spec)]

k = len(ARMS)
def order(b):   # rotation by a step co-prime with k, reversed on odd blocks: every arm visits early and late positions
    o = [ARMS[(i + 4 * b) % k] for i in range(k)]
    return o[::-1] if b % 2 else o
plan = [dict(cell=f'{t}_b25', cond='preserve', block=b, order=order(b)) for t in tasks for b in range(6)]
ns['execute']('../v30/timing', plan, argv, dry='--dry-run' in sys.argv, meta=dict(arms=ARMS, eq_budgets=EQ, p4_selection=p4sel))
