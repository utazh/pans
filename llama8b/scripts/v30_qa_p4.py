"""v30 QuALITY with selection period P4: RAPS-P4 readers for the 12 formal articles + evaluation on all real questions.
Recipe identical to v29_quality_pipeline.py `readers` (synthetic calibration questions syn_fit/syn_val, 64/64; v26 geometry
rule restricted to 4-layer windows; R4 training 200 steps, eval 25, seed 24; frozen R0; G20/G24 by the frozen label-free
revert rule), only the selection period changes 8 -> 4 (as in v30_p4_readers.py, incl. the largest-feasible fallback when a
4-layer window admits fewer than 12 consumers). No dev selection (fixed-degree report, as the v29 QuALITY study).
Eval arms per article: RAPS-P4 G24/G20/G16 (or the feasible set) and NS-P4 at the byte-aligned 21.43/17.86/14.29 %.
Usage: v30_qa_p4.py <gpu,gpu,...>"""
from pathlib import Path
import json, os, runpy, subprocess, sys, threading, time, queue
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY, MODEL = C['R'], C['PY'], C['MODEL']; sys.path.insert(0, str(R / 'src'))
SETS, SLOTS = ['f0', 'f1', 'f2'], C['TASKS']
EQ = [100 * 24 / 28 / 4, 100 * 20 / 28 / 4, 100 * 16 / 28 / 4]
tag = lambda b: f'{b:.4f}'.rstrip('0').rstrip('.').replace('.', 'p')
W = R / 'studies/v30/qa_p4'; (W / 'logs').mkdir(parents=True, exist_ok=True)
gpus = sys.argv[1].split(','); sha = C['sha']
from contiguous_fuxian.sharedkv_v24.records import load, FIELDS, PLAN_FORMAT, assert_plan
from contiguous_fuxian.sharedkv_v21.bundles import isolate
from contiguous_fuxian.sharedkv_v27.frozen_control import export_frozen_r0
import torch

def geometry_graph_p4(row, period=4, need=12):   # v29_subj_p4.py rule with `need` parameter (see v30_p4_readers.py)
    sources = list(range(28)); target = need
    for start in sorted(range(0, 28, period), reverse=True):
        remaining = list(range(start + 1, min(start + period, 28))); groups = {}
        for i in remaining: groups.setdefault(tuple(row['selection_by_layer'][i]), []).append(i)
        for group in sorted(groups.values(), key=lambda g: (-len(g), -max(g))):
            if need <= 0: break
            n = min(need, len(group) - 1)
            if n <= 0: continue
            chosen = group[-(n + 1):]; src = chosen[len(chosen) // 2]
            for i in chosen: sources[i] = src
            need -= n
    if need: raise ValueError(f'P4 geometry admits only {target - need} consumers')
    return sources
def feasible(row):
    for c in range(12, 0, -1):
        try: geometry_graph_p4(row, need=c); return c
        except ValueError: pass
    return 0

def sh(name, argv, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24')
    t0 = time.time()
    with (W / 'logs' / f'{name}.log').open('a') as f:
        rc = subprocess.run(['taskset', '-c', C['CORES'][gpu]] + argv, cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    with (W / 'status.jsonl').open('a') as f: f.write(json.dumps(dict(name=name, gpu=gpu, returncode=rc, seconds=round(time.time() - t0, 1), t=time.time())) + '\n')
    print(name, rc, flush=True)
    if rc: raise RuntimeError(f'{name} rc={rc}')

def article(s, slot, gpu):
    store, index, pc = R / f'runtime_data/v29_quality_{s}_store', R / f'assets/v29_quality_{s}_index', R / f'runtime_data/v29_quality_{s}_pcache'
    D = R / f'data/v29/quality/{s}'; TAG = f'v30q_{s}_{slot}_b25_p4'; A = W / f'{s}_{slot}'; A.mkdir(exist_ok=True)
    rd = A / 'readers'
    if not (A / 'readers_done.json').exists():
        td = {}
        for split, role in (('fit', 'fit'), ('val', 'validation')):
            rec = R / f'runtime_data/{TAG}_teacher_{split}'; td[split] = rec
            if rec.exists(): continue
            bundle = D / f'syn_{split}'; n = sum(1 for l in (bundle / f'{slot}.jsonl').read_text().splitlines() if l.strip())
            a = C['chain'](A / f'capture_{split}', task=slot, budget=25, period=4, samples=n, bundle=bundle, pcache=pc, store=store, index=index)
            a[a.index('--admission') + 1] = 'original'; a.insert(a.index('--mode'), '--diagnostic')
            sh(f'{s}_{slot}_capture_{split}', [PY, '-u', 'scripts/collect_teacher_v26.py', '--teacher-records', str(rec), '--split', role, '--', *a], gpu)
        fit, val = load(td['fit'], slot, 'fit'), load(td['val'], slot, 'validation'); row = next(iter(fit.values()))
        assert row['period'] == 4
        c_max = feasible(row); sources = geometry_graph_p4(row, need=min(12, c_max))
        for item in list(fit.values()) + list(val.values()): assert_plan(item, sources)
        plan_p = R / f'configs/{TAG}_plan.json'
        if not plan_p.exists():
            plan = dict(format=PLAN_FORMAT, **{k: row[k] for k in FIELDS}, sources=sources, unique_sources=len(set(sources)),
                        design_query_hashes=sorted(fit), status=f'v30_quality_p4_geometry_only', feasible_consumers=c_max, quality_certified=False)
            with plan_p.open('x') as f: json.dump(plan, f, indent=2)
        ck = R / f'assets/{TAG}/{slot}.pt'
        if not ck.exists():
            sh(f'{s}_{slot}_train', [PY, '-u', 'scripts/train_instrumented_v25.py', '--native-blocks', '4', '--worklog', str(A / 'train_compute.json'), '--',
               '--model-path', MODEL, '--store-root', str(store), '--task', slot, '--teacher-fit', str(td['fit']), '--teacher-val', str(td['val']),
               '--plan', str(plan_p), '--steps', '200', '--eval-every', '25', '--seed', '24', '--device', 'cuda', '--output', str(ck)], gpu)
        top = f'G{28 - min(12, c_max)}'
        r0 = rd / top / f'{slot}.pt'; r0.parent.mkdir(parents=True, exist_ok=True)
        if not r0.exists(): export_frozen_r0(ck, r0)
        b = torch.load(r0, map_location='cpu', weights_only=False); src = b['sources']
        cons = [l for l in range(28) if src[l] != l]; order = sorted(cons, key=lambda l: (-abs(l - src[l]), l)); names = [top]
        for name, kept in (('G20', 8), ('G24', 4)):
            if kept >= len(cons): continue
            child = isolate(b, sorted(set(cons) - set(order[:len(cons) - kept]))); names.append(name)
            p = rd / name / f'{slot}.pt'; p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                with p.open('xb') as f: torch.save(child, f)
        (A / 'readers_done.json').write_text(json.dumps(dict(candidates=names, feasible_consumers=c_max, sources=sources, revert_order=order,
            plan_sha256=sha(plan_p), checkpoint_sha256=sha(ck), readers={n: sha(rd / n / f'{slot}.pt') for n in names}), indent=1))
    names = json.loads((A / 'readers_done.json').read_text())['candidates']
    bundle = D / 'eval'; n = sum(1 for l in (bundle / f'{slot}.jsonl').read_text().splitlines() if l.strip())
    arms = [(f'RAPSP4_{g}', dict(budget=25, adapters=rd / g), False) for g in names] + [(f'NSP4_{tag(b)}', dict(budget=b), True) for b in EQ]
    for name, spec, ext in arms:
        o = A / 'eval' / name
        if (o / 'summary.json').exists(): continue
        assert not o.exists() and not list(o.parent.glob(f'{name}.*')), f'partial run {o}; move aside first'
        o.parent.mkdir(exist_ok=True)
        sh(f'{s}_{slot}_eval_{name}', [PY, C['entry'](ext), *C['chain'](o, task=slot, period=4, samples=n, bundle=bundle, pcache=pc, store=store, index=index, **spec)], gpu)

work = queue.Queue(); [work.put((s, slot)) for s in SETS for slot in SLOTS]; errs = []
def worker(gpu):
    while True:
        try: s, slot = work.get_nowait()
        except queue.Empty: return
        try: article(s, slot, gpu)
        except Exception as e: errs.append(f'{s}_{slot}: {e!r}'); print('FAILED', s, slot, repr(e), flush=True)
ts = [threading.Thread(target=worker, args=(g,)) for g in gpus]; [t.start() for t in ts]; [t.join() for t in ts]
(W / 'done.json').write_text(json.dumps(dict(errors=errs, finished=time.time()))); print('V30_QA_P4_DONE', errs, flush=True)
