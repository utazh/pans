"""v34 baseline quality on the frozen test sets (any model tree via V30_ROOT / V30_MODEL / PANS_* env): ContiguousKV, IMPRESS,
AS+H2O+LRU at 25 %, through the unchanged v26 chain with v21 --mode original (CKV / IMPRESS: scripts/v29_baseline_entry.py,
AS: scripts/v33_as_entry.py). Usage: v34_baseline_quality.py <gpu,gpu,...> <impress_c64_dir> <impress_manifest> <as_c64_dir>"""
from pathlib import Path
import runpy, sys
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY, MODEL = C['R'], C['PY'], C['MODEL']
gpus = sys.argv[1].split(','); C64, MAN, CAS = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
W = R / 'studies/v34/baselines'; W.mkdir(parents=True, exist_ok=True)
queues = {g: [] for g in gpus}; i = 0
for m, kv, method, entry in (('CKV', C['PCACHE'][gpus[0]], ['--method', 'contigkv'], 'v29_baseline_entry.py'),
                             ('IMP', C64, ['--method', 'impress', '--impress-reorder', str(MAN)], 'v29_baseline_entry.py'),
                             ('AS', CAS, ['--method', 'as_h2o_lru'], 'v33_as_entry.py')):
    for t in C['TASKS']:
        g = gpus[i % len(gpus)]; i += 1; o = W / 'runs' / f'{m}_test_{t}'
        queues[g].append(dict(name=f'{m}_test_{t}', argv=[PY, str(R / 'scripts' / entry), '--admission', 'original', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', MODEL, '--store-root', str(R / 'runtime_data/prefix_store'), '--', '--pcache-root', str(kv), '--', '--mode', 'original', '--',
            *method, '--task', t, '--budget', '25', '--max-tokens', '1', '--warmup', '0', '--samples', str(C['N_TEST'][t]),
            '--bundle-dir', str(C['test_bundle'](t)), '--output', str(o)]))
C['run_queues'](W, queues)
for m in ('CKV', 'IMP', 'AS'):
    for t in C['TASKS']:
        p = W / 'runs' / f'{m}_test_{t}'
        if (p / 'summary.json').exists():
            rows, s = C['load_run'](p); print(m, t, round(100 * s['accuracy'], 2), round(s['mean_physical_prefetch_kv_bytes'] / 1e6, 1), 'MB', flush=True)
print('BASELINE_QUALITY_DONE', flush=True)
