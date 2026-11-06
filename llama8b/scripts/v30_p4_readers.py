"""v30 P4 readers for SST-2 / TREC / RTE at 25% (SUBJ already exists in studies/v29/subj_p4, built by the same recipe).
Exactly the v29_subj_p4.py recipe with the task as a parameter: teacher capture with selection period P4 on the frozen
fit64 / validation64 sets (data/v26/p0/{fit,val}) -> G16 plan from the v26 geometry rule with P4 windows (selection
structure only) -> R4 training (200 steps, eval every 25, seed 24) -> frozen R0 -> G20/G24 via the frozen label-free
revert rule (descending |l - s(l)|, ties by layer). Writes a signature file of all inputs next to the readers.
If the P4 geometry admits fewer than 12 consumers (SST-2: the exact 25% budget alternates 60/59 chunks between layers, so
inside a 4-layer window only same-parity layers have identical selections), --max-feasible trains the largest feasible graph
G(28-c) instead and derives the remaining candidates G24 / G20 from it with the same revert rule (only those with fewer
consumers than c). The candidate set is recorded in readers_done.json.
Usage: v30_p4_readers.py <task> <gpu> <cores> [--max-feasible]"""
from pathlib import Path
import hashlib, json, os, runpy, subprocess, sys, time
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY, MODEL = C['R'], C['PY'], C['MODEL']; sys.path.insert(0, str(R / 'src'))
task, gpu, cores = sys.argv[1], sys.argv[2], sys.argv[3]
import os
STEPS = os.environ.get('V30_STEPS', '200')   # Llama-3.1-8B: 800 (the A100 v29m protocol for non-7B models)
CANON = os.environ.get('V30_ROOT') is not None and task == 'subj'   # second-model trees: SUBJ uses canonical fit/val (as v29m_pipeline)
assert task in ('sst2', 'trec', 'rte') or CANON
TAG = f'v30_p0_{task}_b25_p4'; W = R / f'studies/v30/p4_readers/{task}'
sha = C['sha']

def chain(out, mode, bundle, samples):
    return C['chain'](out, task=task, budget=25, period=4, samples=samples, bundle=bundle, pcache=C['PCACHE'][gpu])

def sh(name, argv):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24')
    (W / 'logs').mkdir(parents=True, exist_ok=True); t0 = time.time()
    with (W / 'logs' / f'{name}.log').open('x') as f:
        rc = subprocess.run(['taskset', '-c', cores] + argv, cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    with (W / 'status.jsonl').open('a') as f: f.write(json.dumps(dict(name=name, returncode=rc, seconds=round(time.time() - t0, 1))) + '\n')
    print(name, rc, flush=True)
    if rc: raise SystemExit(name)

MAXF = '--max-feasible' in sys.argv
def n_layers(row): return len(row['selection_by_layer'])
def target_consumers(L): return 2 * round(L * 3 / 14)   # A100 v29m rule (geometry_map frac 3/7): 12 for 28 layers, 14 for 32
def geometry_graph_p4(row, period=4, need=None):   # scripts/v29_subj_p4.py rule; layer count from the record, `need` parameter added
    L = n_layers(row); need = target_consumers(L) if need is None else need
    sources = list(range(L)); target = need
    for start in sorted(range(0, L, period), reverse=True):
        remaining = list(range(start + 1, min(start + period, L))); groups = {}
        for i in remaining: groups.setdefault(tuple(row['selection_by_layer'][i]), []).append(i)
        for group in sorted(groups.values(), key=lambda g: (-len(g), -max(g))):
            if need <= 0: break
            n = min(need, len(group) - 1)
            if n <= 0: continue
            chosen = group[-(n + 1):]; src = chosen[len(chosen) // 2]
            for i in chosen: sources[i] = src
            need -= n
    if need: raise SystemExit(f'P4 geometry admits only {target - need} consumers; G16 infeasible under P4')
    return sources

def feasible_consumers(row, period=4):
    for c in range(target_consumers(n_layers(row)), 0, -1):
        try: geometry_graph_p4(row, period, c); return c
        except SystemExit: pass
    return 0

W.mkdir(parents=True, exist_ok=MAXF)
from contiguous_fuxian.sharedkv_v24.records import load, FIELDS, PLAN_FORMAT, assert_plan
td = {}
for split, role, folder in (('fit', 'fit', 'canonical_fit' if CANON else 'fit'), ('val', 'validation', 'canonical_val' if CANON else 'val')):
    rec = R / f'runtime_data/{TAG}_teacher_{split}'; out = W / f'capture_{split}'
    if MAXF and rec.exists():   # completed capture of the first (infeasible-G16) attempt: identity capture, reused as is
        td[split] = rec; continue
    a = chain(out, 'identity', R / f'data/v26/p0/{folder}', 64)
    a[a.index('--admission') + 1] = 'original'
    v21 = a.index('--mode'); a.insert(v21, '--diagnostic')
    sh(f'capture_{split}', [PY, '-u', 'scripts/collect_teacher_v26.py', '--teacher-records', str(rec), '--split', role, '--', *a])
    td[split] = rec
fit, val = load(td['fit'], task, 'fit'), load(td['val'], task, 'validation'); row = next(iter(fit.values()))
assert row['period'] == 4, row['period']
c_max = feasible_consumers(row); LAY = n_layers(row); TGT = target_consumers(LAY)
if c_max < TGT and not MAXF: raise SystemExit(f'P4 geometry admits only {c_max} consumers; rerun with --max-feasible')
sources = geometry_graph_p4(row, need=min(TGT, c_max))
for item in list(fit.values()) + list(val.values()): assert_plan(item, sources)
plan = dict(format=PLAN_FORMAT, **{k: row[k] for k in FIELDS}, sources=sources, unique_sources=len(set(sources)),
            design_query_hashes=sorted(fit), status=f'v30_{task}_p4_G{LAY - min(TGT, c_max)}_geometry_only', feasible_consumers=c_max, quality_certified=False,
            geometry_checked_on=dict(fit=sorted(fit), validation=sorted(val)))
plan_p = R / f'configs/{TAG}_plan.json'
with plan_p.open('x') as f: json.dump(plan, f, indent=2)
print('P4 sources', sources, flush=True)
ck = R / f'assets/{TAG}/{task}.pt'
sh('train', [PY, '-u', 'scripts/train_instrumented_v25.py', '--native-blocks', '4', '--worklog', str(W / 'train_compute.json'), '--',
   '--model-path', MODEL, '--store-root', str(R / 'runtime_data/prefix_store'), '--task', task, '--teacher-fit', str(td['fit']),
   '--teacher-val', str(td['val']), '--plan', str(plan_p), '--steps', STEPS, '--eval-every', '25', '--seed', '24', '--device', 'cuda',
   '--output', str(ck)])
import torch
from contiguous_fuxian.sharedkv_v27.frozen_control import export_frozen_r0
from contiguous_fuxian.sharedkv_v21.bundles import isolate
top = f'G{LAY - min(TGT, c_max)}'   # name = number of distinct banks (sources) kept
r0 = W / f'readers/{top}' / f'{task}.pt'; r0.parent.mkdir(parents=True); export_frozen_r0(ck, r0)
b = torch.load(r0, map_location='cpu', weights_only=False); s = b['sources']
cons = [l for l in range(len(s)) if s[l] != l]; order = sorted(cons, key=lambda l: (-abs(l - s[l]), l))
names = [top]
if LAY == 28:   # Qwen2.5-7B: unchanged v29 candidate set (G20 keeps 8 consumers, G24 keeps 4)
    plan_deg = [(name, len(cons) - kept) for name, kept in (('G20', 8), ('G24', 4)) if kept < len(cons)]
else:           # other depths: the A100 v29m S2 / S1 rule (revert round(c/3) / round(2c/3) consumers)
    plan_deg = [(f'G{LAY - (len(cons) - k)}', k) for k in (round(len(cons) / 3), round(2 * len(cons) / 3))]
for name, k in plan_deg:
    names.append(name)
    child = isolate(b, sorted(set(cons) - set(order[:k])))
    p = W / f'readers/{name}/{task}.pt'; p.parent.mkdir(parents=True)
    with p.open('xb') as f: torch.save(child, f)
sig = dict(task=task, budget=25, period=4, model=MODEL, steps=int(STEPS), seed=24, eval_every=25, sources=sources, revert_order=order,
           plan=str(plan_p), plan_sha256=sha(plan_p), checkpoint=str(ck), checkpoint_sha256=sha(ck),
           fit=str(R / 'data/v26/p0' / ('canonical_fit' if CANON else 'fit') / f'{task}.jsonl'),
           val=str(R / 'data/v26/p0' / ('canonical_val' if CANON else 'val') / f'{task}.jsonl'),
           readers={n: dict(path=str(W / f'readers/{n}/{task}.pt'), sha256=sha(W / f'readers/{n}/{task}.pt')) for n in names}, candidates=names, feasible_consumers=c_max)
(W / 'readers_done.json').write_text(json.dumps(sig, indent=1)); print('P4_READERS_DONE', task, flush=True)
