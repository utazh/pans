import json, runpy, os
from pathlib import Path
os.environ['V30_ROOT'] = '/data1/pans_l8/raps_l8'
C = runpy.run_path('/home/panzihang/src/prism_layerkv/scripts/v30_common.py'); R = C['R']
Q = R / 'studies/v30/quality/runs'; B = R / 'studies/v34/baselines/runs'
sel = {c: v['selected'] for c, v in json.loads((R / 'results/v30/p4_selection.json').read_text())['cells'].items()}
for t in C['TASKS']:
    s = sel[t + '_b25']; out = {}
    arms = {'NS8': Q / f'ns_p8_test_{t}_k25', 'NS4': Q / f'ns_p4_test_{t}_k25',
            'RAPS4': Q / (f'p4_test_{s}_{t}_k25' if s != 'I' else f'ns_p4_test_{t}_k25'),
            'CKV': B / f'CKV_test_{t}', 'IMP': B / f'IMP_test_{t}', 'AS': B / f'AS_test_{t}',
            'CAKE': Q / f'cake_static_{t}_k25', 'LAVA': Q / f'lava_static_{t}_k25'}
    for d in sorted(Q.glob(f'p4_test_*_{t}_k25')): arms['all_' + d.name.split('_')[2]] = d
    for k in ('14p2857', '17p8571', '21p4286'): arms[f'NS4_{k}'] = Q / f'ns_eq_{t}_k{k}_p4'
    for a, p in arms.items():
        if (p / 'summary.json').exists():
            r, x = C['load_run'](p); out[a] = (round(100 * x['accuracy'], 2), round(x['mean_physical_prefetch_kv_bytes'] / 1e6, 1))
    print(t, 'selected', s, json.dumps(out))
