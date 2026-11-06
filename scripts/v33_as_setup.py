"""v33 AS+H2O+LRU on the 3090 (Qwen2.5-7B, 25 %).
runner : scripts/run_round4_methods_as.py = byte copy of the A100 runner (backup ~/src/pans/src/scripts/run_round4_methods.py,
         adds --method as_h2o_lru) with only the default model path switched to the 3090 path.
build  : c64 plain Pcache runtime_data/v33_pcache_as_c64_ssd via --build-kv (as xfer/as_smoke.sh on the A100).
quality: frozen test sets of the four tasks through scripts/v33_as_entry.py (unchanged v26 chain, v21 --mode original).
Usage: v33_as_setup.py runner | build <gpu> | quality <gpu,gpu,...>"""
from pathlib import Path
import hashlib, json, os, runpy, subprocess, sys
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY, MODEL = C['R'], C['PY'], C['MODEL']
CAS = R / 'runtime_data/v33_pcache_as_c64_ssd'; W = R / 'studies/v33/as'; W.mkdir(parents=True, exist_ok=True)
step = sys.argv[1]
if step == 'runner':
    src = Path('/home/panzihang/src/pans/src/scripts/run_round4_methods.py'); dst = R / 'scripts/run_round4_methods_as.py'
    t = src.read_text(); a = "'/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct'"; assert t.count(a) == 1
    if not dst.exists(): dst.write_text(t.replace(a, f"'{MODEL}'"))
    (W / 'runner.json').write_text(json.dumps(dict(source=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                                                   dst_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(), change='default model path only'), indent=1))
    print('RUNNER_READY', flush=True)
elif step == 'build':
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=sys.argv[2], PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24',
               PANS_MODEL_PATH=MODEL)
    if not (CAS / '.contiguous_fuxian_complete').exists():
        with (W / 'build.log').open('a') as f:
            rc = subprocess.run([PY, 'scripts/run_round4_methods_as.py', '--method', 'as_h2o_lru', '--build-kv', '--kv-dir', str(CAS), '--task', 'sst2',
                                 '--budget', '25', '--samples', '2', '--warmup', '0', '--bundle-dir', str(R / 'studies/v28_frozen/bundles/sst2_b25/eval'),
                                 '--output', str(W / 'build_run')], cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc: raise SystemExit(f'AS build failed rc={rc}')
    print('BUILD_DONE', sorted(p.name for p in CAS.iterdir())[:3], sum(1 for _ in CAS.iterdir()), flush=True)
else:
    gpus = sys.argv[2].split(','); queues = {g: [] for g in gpus}
    for i, t in enumerate(C['TASKS']):
        g = gpus[i % len(gpus)]; o = W / 'runs' / f'as_test_{t}'
        queues[g].append(dict(name=f'as_test_{t}', argv=[PY, str(R / 'scripts/v33_as_entry.py'), '--admission', 'original', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', MODEL, '--store-root', str(R / 'runtime_data/prefix_store'), '--', '--pcache-root', str(CAS), '--',
            '--mode', 'original', '--', '--method', 'as_h2o_lru', '--task', t, '--budget', '25', '--max-tokens', '1', '--warmup', '0',
            '--samples', str(C['N_TEST'][t]), '--bundle-dir', str(C['test_bundle'](t)), '--output', str(o)]))
    C['run_queues'](W, queues)
    for t in C['TASKS']:
        rows, s = C['load_run'](W / 'runs' / f'as_test_{t}'); print(t, round(100 * s['accuracy'], 2), round(s['mean_physical_prefetch_kv_bytes'] / 1e6, 1), 'MB', flush=True)
    print('QUALITY_DONE', flush=True)
