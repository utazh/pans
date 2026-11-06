"""v30 QuALITY (12 formal articles = sets f0/f1/f2 x 4 slots, all real questions, 25% family), reusing the v29 QuALITY
stores / Pcaches / K4 indexes unchanged. New arms per article (existing v29 arms I, FULLKV, syn_G24/G20/G16 are reused):
  NSP4_25            identity, selection period 4, budget 25
  NS_<b>             identity P8 at the byte-aligned budgets 21.43 / 17.86 / 14.29 (= G24 / G20 / G16 bytes)
  CAKE_<b> / LAVA_<b> prefix-window static selections at 25 and the three aligned budgets
Steps: prefix statistics (v30_kvscore.py) -> selections (v30_select_build.py) -> evaluation on GPUs.
Usage: v30_quality_qa.py <gpu,gpu,...>"""
from pathlib import Path
import json, os, runpy, subprocess, sys
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY = C['R'], C['PY']; SLOTS = C['TASKS']; SETS = ['f0', 'f1', 'f2']
EQ = [100 * 24 / 28 / 4, 100 * 20 / 28 / 4, 100 * 16 / 28 / 4]
tag = lambda b: f'{b:.4f}'.rstrip('0').rstrip('.').replace('.', 'p')
gpus = sys.argv[1].split(',')
OUT = R / 'studies/v30/quality_qa'; (OUT / 'logs').mkdir(parents=True, exist_ok=True)
store = lambda s: R / f'runtime_data/v29_quality_{s}_store'
bundle = lambda s: R / f'data/v29/quality/{s}/eval'

# 1) statistics + selections (sequential on the first GPU; skipped when present)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0], PYTHONDONTWRITEBYTECODE='1')
for s in SETS:
    for slot in SLOTS:
        e = dict(env, V30_STORE=str(store(s)), V30_BUNDLE=str(bundle(s)), V30_TAG=f'q{s}_')
        if not (R / f'runtime_data/v30_kvscore/prefix_q{s}_{slot}.npz').exists():
            with (OUT / 'logs/kvscore.log').open('a') as f:
                subprocess.run([PY, 'scripts/v30_kvscore.py', 'prefix', slot], cwd=R, env=e, stdout=f, stderr=subprocess.STDOUT, check=True)
        for m in ('cake', 'lava'):
            for b in (25, *EQ):
                with (OUT / 'logs/select_build.log').open('a') as f:
                    subprocess.run([PY, 'scripts/v30_select_build.py', 'static', m, slot, repr(float(b))], cwd=R, env=e, stdout=f, stderr=subprocess.STDOUT, check=True)
print('STATS_AND_SELECTIONS_READY', flush=True)

# 2) evaluation
jobs = []
for s in SETS:
    for slot in SLOTS:
        n = sum(1 for l in (bundle(s) / f'{slot}.jsonl').read_text().splitlines() if l.strip())
        common = dict(task=slot, samples=n, bundle=bundle(s), store=store(s), index=R / f'assets/v29_quality_{s}_index')
        jobs.append((f'{s}_{slot}_NSP4_25', dict(common, budget=25, period=4), False))
        for b in EQ: jobs.append((f'{s}_{slot}_NS_{tag(b)}', dict(common, budget=b, period=8), True))
        for m in ('cake', 'lava'):
            for b in (25, *EQ):
                sel = R / f'configs/v30_sel/{m}_prefix_q{s}_{slot}_k{tag(b)}.json'
                jobs.append((f'{s}_{slot}_{m.upper()}_{tag(b)}', dict(common, budget=b, period=8, ext=['--static-selection', str(sel)]), True))
queues = {g: [] for g in gpus}
for i, (name, spec, ext) in enumerate(jobs):
    g = gpus[i % len(gpus)]; o = OUT / 'runs' / name
    queues[g].append(dict(name=name, argv=[PY, C['entry'](ext), *C['chain'](o, pcache=R / f"runtime_data/v29_quality_{name.split('_')[0]}_pcache", **spec)]))
with (OUT / 'plan.jsonl').open('a') as f:
    for g, q in queues.items():
        for j in q: f.write(json.dumps(dict(gpu=g, **j)) + '\n')
print({g: len(q) for g, q in queues.items()}, flush=True)
C['run_queues'](OUT, queues)
print('V30_QUALITY_QA_DONE', flush=True)
