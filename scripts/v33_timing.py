"""v33 final 7B comparison, 3090: every method in the SAME timing blocks, SSD-cold (preserve) and warm (ContiguousKV-style
ordinary cache: no cold reset, warmup 4 requests, then 16 measured requests; all methods keep their own published cache
policy and the SAME quotas: 55 MiB GPU / 131 MiB CPU; AS uses its LRU policy). v29 timing rule rev 3 executor: GPU 2,
cores 28-31,60-63, pre-block gate, telemetry, flagged/failed block rerun once. Cell = <task>_b25, 6 blocks x 16 queries.
Arms: NS8 (our system, no sharing, P8) | NS4 (no sharing, P4) | RAPS4 (dev-selected RAPS-P4; results/v30/p4_selection.json)
      CKV (ContiguousKV) | IMP (IMPRESS, c64 reordered Pcache) | AS (AS+H2O+LRU, c64 plain Pcache)
      CAKE / LAVA (prefix-window static selections, 25 %)
Second model / reruns: env V30_ROOT (tree; its scripts/v29_timing_lib.py is used, env-configured via PANS_TIMING_* / PANS_ROOT),
V33_C16 / V33_C64 / V33_CAS / V33_REORDER (SSD Pcaches, IMPRESS manifest), V33_STUDY (output name under studies/v29/..);
--only cell:cond:block[,..] restricts the plan (e.g. reruns of flagged blocks).
Usage: v33_timing.py [--dry-run] [--conds preserve,warm] [--only sst2_b25:preserve:0,...]"""
from pathlib import Path
import json, os, runpy, sys
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY, MODEL = C['R'], C['PY'], C['MODEL']
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py'))
env = lambda k, d: Path(os.environ[k]) if os.environ.get(k) else d
C16 = env('V33_C16', R / 'runtime_data/v29_pcache_gpu2_ssd'); C64 = env('V33_C64', R / 'runtime_data/v29_pcache_impress_c64_ssd')
CAS = env('V33_CAS', R / 'runtime_data/v33_pcache_as_c64_ssd'); REORDER = env('V33_REORDER', R / 'configs/v29_impress_reorder_p0_fit4.json')
STUDY = os.environ.get('V33_STUDY', '../v33/timing')
p4sel = json.loads((R / 'results/v30/p4_selection.json').read_text())['cells']
P4R = {t: R / f'studies/v30/p4_readers/{t}/readers' for t in ('sst2', 'subj', 'trec', 'rte')}
if not os.environ.get('V30_ROOT'): P4R['subj'] = R / 'studies/v29/subj_p4/readers'
ARMS = ['NS8', 'NS4', 'RAPS4', 'CKV', 'IMP', 'AS', 'CAKE', 'LAVA']
args = sys.argv[1:]
CONDS = args[args.index('--conds') + 1].split(',') if '--conds' in args else ['preserve', 'warm']

def argv(out, cell, arm, cond, block, name):
    task = cell.split('_')[0]; o = out / 'runs' / name; bundle = R / f'studies/v28_frozen/bundles/{cell}/b{block:02d}'
    if arm in ('CKV', 'IMP', 'AS'):
        kv = {'CKV': C16, 'IMP': C64, 'AS': CAS}[arm]
        cold = ['--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(kv), '--cold-log', f'{o}.cold.jsonl'] if cond == 'preserve' else []
        method = {'CKV': ['--method', 'contigkv'], 'IMP': ['--method', 'impress', '--impress-reorder', str(REORDER)], 'AS': ['--method', 'as_h2o_lru']}[arm]
        entry = R / ('scripts/v33_as_entry.py' if arm == 'AS' else 'scripts/v29_baseline_entry.py')
        return [PY, str(entry), '--admission', 'original', '--audit-jsonl', f'{o}.v26.jsonl', '--',
                '--model-path', MODEL, '--store-root', str(R / 'runtime_data/prefix_store'), '--', '--pcache-root', str(kv), *cold, '--',
                '--mode', 'original', '--', *method, '--task', task, '--budget', '25', '--max-tokens', '1', '--warmup', '4', '--samples', '16',
                '--bundle-dir', str(bundle), '--output', str(o)]
    spec, ext = dict(budget=25, period=8), False
    if arm == 'NS4': spec['period'] = 4
    elif arm == 'RAPS4':
        s = p4sel[cell]['selected']; spec['period'] = 4
        if s != 'I': spec['adapters'] = P4R[task] / s
    elif arm in ('CAKE', 'LAVA'):
        spec['ext'] = ['--static-selection', str(R / f'configs/v30_sel/{arm.lower()}_prefix_{task}_k25.json')]; ext = True
    return [PY, C['entry'](ext), *C['chain'](o, task=task, samples=16, warmup=4, cold=cond if cond == 'preserve' else None, pcache=C16, bundle=bundle, **spec)]

k = len(ARMS)
def order(b):   # each arm visits early and late positions across the 6 blocks
    o = [ARMS[(i + 3 * b) % k] for i in range(k)]
    return o[::-1] if b % 2 else o
plan = [dict(cell=f'{t}_b25', cond=c, block=b, order=order(b)) for c in CONDS for t in C['TASKS'] for b in range(6)]
if '--only' in args:
    keep = {tuple(x.split(':')) for x in args[args.index('--only') + 1].split(',')}
    plan = [p for p in plan if (p['cell'], p['cond'], str(p['block'])) in keep]
ns['execute'](STUDY, plan, argv, dry='--dry-run' in args, meta=dict(arms=ARMS, p4_selection={c: v['selected'] for c, v in p4sel.items()},
              cache='CKLFU 55 MiB GPU / 131 MiB CPU for all CKLFU systems; AS: LRU, same quotas', warmup=4))
