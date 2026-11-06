"""v30 smoke 2: fixed CAKE selection and LAVa layer-budget profile on 16 TREC dev queries at 25%; checks that the
engine accepts them, that the selected chunks per layer are exactly the requested ones, and that bytes equal NS-25.
Usage: v30_smoke2.py <gpu>"""
import json, runpy, sys
from pathlib import Path
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY = C['R'], C['PY']; g = sys.argv[1]
OUT = R / 'studies/v30/smoke'; pc = C['PCACHE'][g]
def job(name, ext):
    o = OUT / 'runs' / name
    return dict(name=name, argv=[PY, C['entry'](True), *C['chain'](o, task='trec', budget=25, period=8, samples=16, bundle=C['DEV'], pcache=pc, ext=ext)])
C['run_queues'](OUT, {g: [job('cake_static_b25', ['--static-selection', str(R / 'configs/v30_sel/cake_prefix_trec_k25.json')]),
                          job('lava_budget_b25', ['--layer-budget-profile', str(R / 'configs/v30_sel/lavabudget_trec_k25.json')])]})
base, tb = C['load_run'](OUT / 'runs/base_b25')
for n in ('cake_static_b25', 'lava_budget_b25'):
    rows, t = C['load_run'](OUT / f'runs/{n}')
    print(n, json.dumps(dict(acc=t['accuracy'], bytes=t['mean_physical_prefetch_kv_bytes'], bytes_over_ns=t['mean_physical_prefetch_kv_bytes'] / tb['mean_physical_prefetch_kv_bytes'],
                             paired=C['paired'](base, rows))))
