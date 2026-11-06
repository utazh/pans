"""v30 QuALITY SSD-cold timing (v29 timing rule rev 3 executor: GPU 2, cores 28-31,60-63, gate, telemetry, one rerun of
flagged/failed blocks). One block per article (12), all real questions, warmup 4, rotated arm orders. Arms:
  NS25        identity P8 25 %          (link to the v29 QuALITY study)
  NSP4_25     identity P4 25 %
  NSP4_21/18/14  identity P4 at the byte-aligned 21.43 / 17.86 / 14.29 %
  RAPSP4_<G>  RAPS-P4 readers of studies/v30/qa_p4 (G24/G20/G16 or the feasible set)
  CAKE / LAVA prefix-window static selections at 25 % (P8 engine, as in the quality runs)
Usage: v30_timing_qa.py [--dry-run]"""
from pathlib import Path
import json, runpy, sys
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY = C['R'], C['PY']
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py'))
SETS, SLOTS = ['f0', 'f1', 'f2'], C['TASKS']
EQ = {'NSP4_21': 100 * 24 / 28 / 4, 'NSP4_18': 100 * 20 / 28 / 4, 'NSP4_14': 100 * 16 / 28 / 4}
def n_rows(p): return sum(1 for l in Path(p).read_text().splitlines() if l.strip())

def spec(s, slot, arm):
    if arm == 'NS25': return dict(budget=25, period=8), False
    if arm == 'NSP4_25': return dict(budget=25, period=4), False
    if arm in EQ: return dict(budget=EQ[arm], period=4), True
    if arm.startswith('RAPSP4_'): return dict(budget=25, period=4, adapters=R / f'studies/v30/qa_p4/{s}_{slot}/readers/{arm[7:]}'), False
    m = arm.lower()
    return dict(budget=25, period=8, ext=['--static-selection', str(R / f'configs/v30_sel/{m}_prefix_q{s}_{slot}_k25.json')]), True

def argv(out, cell, arm, cond, block, name):
    s, slot = cell.split('_'); sp, ext = spec(s, slot, arm); o = out / 'runs' / name
    pc = R / f'runtime_data/v29_quality_{s}_pcache'; bundle = R / f'data/v29/quality/{s}/eval'
    return [PY, C['entry'](ext), *C['chain'](o, task=slot, samples=n_rows(bundle / f'{slot}.jsonl'), warmup=4, cold=cond, pcache=pc, bundle=bundle,
                                             store=R / f'runtime_data/v29_quality_{s}_store', index=R / f'assets/v29_quality_{s}_index', **sp)]
plan = []
for i, (s, slot) in enumerate([(s, slot) for s in SETS for slot in SLOTS]):
    cands = json.loads((R / f'studies/v30/qa_p4/{s}_{slot}/readers_done.json').read_text())['candidates']
    arms = ['NS25', 'NSP4_25', *EQ, *[f'RAPSP4_{g}' for g in cands], 'CAKE', 'LAVA']; k = len(arms)
    order = [arms[(j + 3 * i) % k] for j in range(k)]
    plan.append(dict(cell=f'{s}_{slot}', cond='preserve', block=0, order=order[::-1] if i % 2 else order))
ns['execute']('../v30/timing_qa', plan, argv, dry='--dry-run' in sys.argv, meta=dict(note='one block per article; rotated orders'))
