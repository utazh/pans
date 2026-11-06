"""v35 QuALITY final timing (7B, 12 formal articles, 25 %): all methods in the same block, SSD-cold and warm (ContiguousKV-style:
no cold reset, warmup 4, same 55/131 MiB quotas; AS: LRU). One block per article, all its real questions. v29 timing rule
rev 3 executor (GPU 2, cores 28-31,60-63, gate, telemetry, one rerun of flagged/failed blocks).
Arms: NS8 (identity P8) | NS4 (identity P4) | RAPS4 (RAPS-P4 G24, the fixed-degree QuALITY setting; readers studies/v30/qa_p4)
      CKV | IMP (per-set reordered c64 Pcache, SSD copy) | AS (per-set c64 plain Pcache, SSD copy) | CAKE | LAVA (prefix window, 25 %)
Usage: v35_timing_qa.py [--dry-run] [--only cell:cond:block,...]"""
from pathlib import Path
import json, runpy, sys
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY, MODEL = C['R'], C['PY'], C['MODEL']
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py'))
SETS, SLOTS = ['f0', 'f1', 'f2'], C['TASKS']; Q = Path('/data1/pans_qa')
ARMS = ['NS8', 'NS4', 'RAPS4', 'CKV', 'IMP', 'AS', 'CAKE', 'LAVA']
args = sys.argv[1:]
def n_rows(p): return sum(1 for l in Path(p).read_text().splitlines() if l.strip())

def argv(out, cell, arm, cond, block, name):
    s, slot = cell.split('_'); o = out / 'runs' / name; bundle = R / f'data/v29/quality/{s}/eval'
    store, index = R / f'runtime_data/v29_quality_{s}_store', R / f'assets/v29_quality_{s}_index'
    n = n_rows(bundle / f'{slot}.jsonl'); c16 = R / f'runtime_data/v29_quality_{s}_pcache'
    if arm in ('CKV', 'IMP', 'AS'):
        kv = {'CKV': c16, 'IMP': R / f'runtime_data/v35_qa_imp_c64_{s}_ssd', 'AS': R / f'runtime_data/v35_qa_as_c64_{s}_ssd'}[arm]
        cold = ['--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(kv), '--cold-log', f'{o}.cold.jsonl'] if cond == 'preserve' else []
        method = {'CKV': ['--method', 'contigkv'], 'IMP': ['--method', 'impress', '--impress-reorder', str(Q / f'impress_reorder_{s}.json')],
                  'AS': ['--method', 'as_h2o_lru']}[arm]
        entry = R / ('scripts/v33_as_entry.py' if arm == 'AS' else 'scripts/v29_baseline_entry.py')
        return [PY, str(entry), '--admission', 'original', '--audit-jsonl', f'{o}.v26.jsonl', '--', '--model-path', MODEL, '--store-root', str(store), '--',
                '--pcache-root', str(kv), *cold, '--', '--mode', 'original', '--', *method, '--task', slot, '--budget', '25', '--max-tokens', '1',
                '--warmup', '4', '--samples', str(n), '--bundle-dir', str(bundle), '--output', str(o)]
    spec, ext = dict(budget=25, period=8), False
    if arm == 'NS4': spec['period'] = 4
    elif arm == 'RAPS4': spec.update(period=4, adapters=R / f'studies/v30/qa_p4/{s}_{slot}/readers/G24')
    elif arm in ('CAKE', 'LAVA'): spec['ext'] = ['--static-selection', str(R / f'configs/v30_sel/{arm.lower()}_prefix_q{s}_{slot}_k25.json')]; ext = True
    return [PY, C['entry'](ext), *C['chain'](o, task=slot, samples=n, warmup=4, cold=cond if cond == 'preserve' else None, pcache=c16, bundle=bundle,
                                             store=store, index=index, **spec)]
k = len(ARMS); cells = [f'{s}_{slot}' for s in SETS for slot in SLOTS]
def order(i):
    o = [ARMS[(j + 3 * i) % k] for j in range(k)]
    return o[::-1] if i % 2 else o
plan = [dict(cell=c, cond=cond, block=0, order=order(i)) for cond in ('preserve', 'warm') for i, c in enumerate(cells)]
if '--only' in args:
    keep = {tuple(x.split(':')) for x in args[args.index('--only') + 1].split(',')}
    plan = [p for p in plan if (p['cell'], p['cond'], str(p['block'])) in keep]
ns['execute']('../v35/timing_qa', plan, argv, dry='--dry-run' in args, meta=dict(arms=ARMS, raps='RAPS-P4 G24 fixed degree', warmup=4))
