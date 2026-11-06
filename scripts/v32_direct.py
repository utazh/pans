"""v32 final method step: train each sharing degree directly (RAPS-P4, 25 %, the four ContiguousKV tasks).
Until now G20 / G24 were derived from the jointly trained top graph (G16; SST-2: G21) by reverting consumers
(sharedkv_v21.bundles.isolate), which removes the compensation the reverted layers provided. Here the SAME reduced graphs
(frozen label-free revert order: descending |l - s(l)|, ties by layer) get their own readers, trained with the identical
recipe and data as the top graph: P4 teacher records (fit64 / validation64), R4 training 200 steps, eval 25, seed 24,
--native-blocks 4, frozen R0 export. Candidates for the frozen dev rule (rev 3): {I, G24t, G20t, top}.
Usage: v32_direct.py train|dev|test <gpu,gpu,...>  |  report"""
from pathlib import Path
import json, math, os, runpy, subprocess, sys, threading, time, queue
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY, MODEL = C['R'], C['PY'], C['MODEL']; TASKS = C['TASKS']; sys.path.insert(0, str(R / 'src'))
W = R / 'studies/v32/direct'; (W / 'logs').mkdir(parents=True, exist_ok=True)
Q = R / 'studies/v30/quality/runs'
P4R = {t: R / f'studies/v30/p4_readers/{t}/readers' for t in ('sst2', 'trec', 'rte')}; P4R['subj'] = R / 'studies/v29/subj_p4/readers'
TOP = {t: ('G21' if t == 'sst2' else 'G16') for t in TASKS}
TAG = {t: ('v29_p0_subj_b25_p4' if t == 'subj' else f'v30_p0_{t}_b25_p4') for t in TASKS}
step = sys.argv[1]

def degrees(t):
    plan = json.loads((R / f'configs/{TAG[t]}_plan.json').read_text()); s = plan['sources']
    cons = [l for l in range(28) if s[l] != l]; order = sorted(cons, key=lambda l: (-abs(l - s[l]), l)); out = {}
    for name, kept in (('G20t', 8), ('G24t', 4)):
        if kept >= len(cons): continue
        rev = set(order[:len(cons) - kept]); src = [l if l in rev else s[l] for l in range(28)]
        out[name] = dict(sources=src, reverted=sorted(rev))
    return plan, out

def sh(name, argv, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24')
    t0 = time.time()
    with (W / 'logs' / f'{name}.log').open('a') as f:
        rc = subprocess.run(['taskset', '-c', C['CORES'][gpu]] + argv, cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    with (W / 'status.jsonl').open('a') as f: f.write(json.dumps(dict(name=name, gpu=gpu, returncode=rc, seconds=round(time.time() - t0, 1), t=time.time())) + '\n')
    print(name, rc, flush=True)
    if rc: raise RuntimeError(f'{name} rc={rc}')

def train_one(t, name, d, gpu):
    from contiguous_fuxian.sharedkv_v24.records import load, assert_plan
    from contiguous_fuxian.sharedkv_v27.frozen_control import export_frozen_r0
    r0 = W / 'readers' / name / f'{t}.pt'
    if r0.exists(): return
    base, _ = degrees(t)
    fit = load(R / f'runtime_data/{TAG[t]}_teacher_fit', t, 'fit'); val = load(R / f'runtime_data/{TAG[t]}_teacher_val', t, 'validation')
    for item in list(fit.values()) + list(val.values()): assert_plan(item, d['sources'])
    plan = dict(base, sources=d['sources'], unique_sources=len(set(d['sources'])), status=f'v32_{t}_p4_{name}_direct_reduced_graph',
                reverted_from_top=d['reverted'])
    plan_p = R / f'configs/v32_{t}_b25_p4_{name}_plan.json'
    if not plan_p.exists():
        with plan_p.open('x') as f: json.dump(plan, f, indent=2)
    ck = R / f'assets/v32_{t}_b25_p4_{name}/{t}.pt'
    if not ck.exists():
        sh(f'train_{t}_{name}', [PY, '-u', 'scripts/train_instrumented_v25.py', '--native-blocks', '4', '--worklog', str(W / f'train_{t}_{name}.json'), '--',
           '--model-path', MODEL, '--store-root', str(R / 'runtime_data/prefix_store'), '--task', t,
           '--teacher-fit', str(R / f'runtime_data/{TAG[t]}_teacher_fit'), '--teacher-val', str(R / f'runtime_data/{TAG[t]}_teacher_val'),
           '--plan', str(plan_p), '--steps', '200', '--eval-every', '25', '--seed', '24', '--device', 'cuda', '--output', str(ck)], gpu)
    r0.parent.mkdir(parents=True, exist_ok=True); export_frozen_r0(ck, r0)

if step == 'train':
    gpus = sys.argv[2].split(','); work = queue.Queue(); errs = []
    for t in TASKS:
        for name, d in degrees(t)[1].items(): work.put((t, name, d))
    def worker(g):
        while True:
            try: t, name, d = work.get_nowait()
            except queue.Empty: return
            try: train_one(t, name, d, g)
            except Exception as e: errs.append(f'{t}/{name}: {e!r}'); print('FAILED', t, name, repr(e), flush=True)
    ts = [threading.Thread(target=worker, args=(g,)) for g in gpus]; [x.start() for x in ts]; [x.join() for x in ts]
    if errs: raise SystemExit(f'train errors: {errs}')
    print('TRAIN_DONE', flush=True)
elif step in ('dev', 'test'):
    gpus = sys.argv[2].split(','); queues = {g: [] for g in gpus}; i = 0
    for t in TASKS:
        for name in degrees(t)[1]:
            g = gpus[i % len(gpus)]; i += 1; o = W / 'runs' / f'{step}_{t}_{name}'
            bundle = C['DEV'] if step == 'dev' else C['test_bundle'](t); n = 500 if step == 'dev' else C['N_TEST'][t]
            queues[g].append(dict(name=f'{step}_{t}_{name}', argv=[PY, C['entry'](False), *C['chain'](o, task=t, budget=25, period=4, samples=n, bundle=bundle,
                                                                                              pcache=C['PCACHE'][g], adapters=W / 'readers' / name)]))
    C['run_queues'](W, queues); print(step.upper() + '_DONE', flush=True)
else:
    out = {}
    for t in TASKS:
        rows = {}
        for split in ('dev', 'test'):
            ns, nst = C['load_run'](Q / f'ns_p4_{split}_{t}_k25')
            cand = {'I': (ns, nst), TOP[t]: C['load_run'](Q / f'p4_{split}_{TOP[t]}_{t}_k25')}
            for g in ('G24', 'G20'):
                p = Q / f'p4_{split}_{g}_{t}_k25'
                if (p / 'summary.json').exists(): cand[g + '(derived)'] = C['load_run'](p)
            for g in degrees(t)[1]:
                p = W / 'runs' / f'{split}_{t}_{g}'
                if (p / 'summary.json').exists(): cand[g] = C['load_run'](p)
            res = {}
            for g, (r, tt) in cand.items():
                pr = C['paired'](ns, r)
                res[g] = dict(acc=100 * pr['acc'], net=pr['net'], harmed=pr['harmed'], rescued=pr['rescued'], disagree=pr['disagree'],
                              bytes_vs_I=tt['mean_physical_prefetch_kv_bytes'] / nst['mean_physical_prefetch_kv_bytes'] - 1)
            rows[split] = res
        dev = rows['dev']; tol = math.ceil(0.01 * 500)
        names = [g for g in ('I', 'G24t', 'G20t', TOP[t]) if g in dev]
        elig = [g for g in names if g == 'I' or dev[g]['net'] >= -tol]; bmin = min(dev[g]['bytes_vs_I'] for g in elig)
        near = [g for g in elig if (dev[g]['bytes_vs_I'] - bmin) / (1 + bmin) < 0.005]
        shared = lambda g: 0 if g == 'I' else 28 - int(g[1:3])
        pick = sorted(near, key=lambda g: (-dev[g]['net'], shared(g), names.index(g)))[0]
        out[t] = dict(selected=pick, eligible=elig, rows=rows)
    (R / 'results/v32').mkdir(parents=True, exist_ok=True); (R / 'results/v32/direct.json').write_text(json.dumps(out, indent=1))
    for t, o in out.items():
        print(t, 'selected:', o['selected'], 'eligible:', o['eligible'])
        for split in ('dev', 'test'):
            print('  ', split, {g: (round(v['acc'], 2), v['net'], f"{100 * v['bytes_vs_I']:+.1f}%") for g, v in o['rows'][split].items()})
