"""v29 parallel setting: SUBJ/25% with selection period P4 (+ sharing). Main P8 matrix untouched.
readers: teacher capture (P4, identity, existing canonical fit/val 64/64) -> G16 plan with the v26 geometry rule adapted to
         4-layer windows (selection-structure only, no scores/labels) -> R4 training (v26 recipe: 200 steps, eval 25, seed 24)
         -> frozen R0 -> G20/G24 via the frozen label-free revert rule.
eval   : canonical full test set (1003): I_P4, G24_P4, G20_P4, G16_P4.
Usage: v29_subj_p4.py readers | eval   (GPU via CUDA_VISIBLE_DEVICES; eval uses GPUs 0-3)"""
from pathlib import Path
import json, os, subprocess, sys, threading, time
R = Path('/home/shensicheng/qyd/pans/src'); PY = '/home/shensicheng/qyd/pans/venv/bin/python'
MODEL = '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct'; W = R / 'studies/v29/subj_p4'; sys.path.insert(0, str(R / 'src'))
TAG = 'v29_p0_subj_b25_p4'

def chain(out, mode, bundle, samples, adapters=None, extra_v21=(), lead=('--admission', 'reuse-prefetch'), pcache=R / 'runtime_data/v27_pcache_gpu2'):
    m = ['--mode', mode, '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{out}.extra.jsonl', *extra_v21]
    if adapters: m += ['--adapters', str(adapters), '--allow-candidate', '--packed-ops']
    return [*lead, '--audit-jsonl', f'{out}.v26.jsonl', '--', '--model-path', MODEL, '--store-root', str(R / 'runtime_data/prefix_store'),
            '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(pcache), '--', *m, '--',
            '--task', 'subj', '--budget', '25', '--period', '4', '--max-tokens', '1', '--warmup', '0', '--samples', str(samples),
            '--profile', 'none', '--bundle-dir', str(bundle), '--output', str(out)]

def sh(name, argv, gpu='0', cores='0-7,32-39'):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24')
    (W / 'logs').mkdir(parents=True, exist_ok=True); t0 = time.time()
    with (W / 'logs' / f'{name}.log').open('x') as f:
        rc = subprocess.run(['taskset', '-c', cores] + argv, cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    with (W / 'status.jsonl').open('a') as f: f.write(json.dumps(dict(name=name, returncode=rc, seconds=round(time.time() - t0, 1))) + '\n')
    print(name, rc, flush=True)
    if rc: raise SystemExit(name)

def geometry_graph_p4(row, period=4):  # v26 rule with P8 windows replaced by P4 windows; geometry only
    sources = list(range(28)); need = 12
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
    if need: raise SystemExit(f'P4 geometry admits only {12 - need} consumers; G16 infeasible under P4')
    return sources

if sys.argv[1] == 'readers':
    W.mkdir(parents=True, exist_ok=False)
    from contiguous_fuxian.sharedkv_v24.records import load, FIELDS, PLAN_FORMAT, assert_plan
    td = {}
    for split, role, folder in (('fit', 'fit', 'canonical_fit'), ('val', 'validation', 'canonical_val')):
        rec = R / f'runtime_data/{TAG}_teacher_{split}'; out = W / f'capture_{split}'
        sh(f'capture_{split}', [PY, '-u', 'scripts/collect_teacher_v26.py', '--teacher-records', str(rec), '--split', role, '--',
           *chain(out, 'identity', R / f'data/v26/p0/{folder}', 64, extra_v21=('--diagnostic',), lead=('--admission', 'original'))])
        td[split] = rec
    fit, val = load(td['fit'], 'subj', 'fit'), load(td['val'], 'subj', 'validation'); row = next(iter(fit.values()))
    assert row['period'] == 4, row['period']
    sources = geometry_graph_p4(row)
    for item in list(fit.values()) + list(val.values()): assert_plan(item, sources)
    plan = dict(format=PLAN_FORMAT, **{k: row[k] for k in FIELDS}, sources=sources, unique_sources=len(set(sources)),
                design_query_hashes=sorted(fit), status='v29_subj_p4_G16_geometry_only', quality_certified=False)
    plan_p = R / f'configs/{TAG}_plan.json'
    with plan_p.open('x') as f: json.dump(plan, f, indent=2)
    print('P4 sources', sources, flush=True)
    ck = R / f'assets/{TAG}/subj.pt'
    sh('train', [PY, '-u', 'scripts/train_instrumented_v25.py', '--native-blocks', '4', '--worklog', str(W / 'train_compute.json'), '--',
       '--model-path', MODEL, '--store-root', str(R / 'runtime_data/prefix_store'), '--task', 'subj', '--teacher-fit', str(td['fit']),
       '--teacher-val', str(td['val']), '--plan', str(plan_p), '--steps', '200', '--eval-every', '25', '--seed', '24', '--device', 'cuda',
       '--output', str(ck)])
    import torch
    from contiguous_fuxian.sharedkv_v27.frozen_control import export_frozen_r0
    from contiguous_fuxian.sharedkv_v21.bundles import isolate
    r0 = W / 'readers/G16/subj.pt'; r0.parent.mkdir(parents=True); export_frozen_r0(ck, r0)
    b = torch.load(r0, map_location='cpu', weights_only=False); s = b['sources']
    cons = [l for l in range(28) if s[l] != l]; order = sorted(cons, key=lambda l: (-abs(l - s[l]), l))
    for name, k in (('G20', 4), ('G24', 8)):
        child = isolate(b, sorted(set(cons) - set(order[:k])))
        p = W / f'readers/{name}/subj.pt'; p.parent.mkdir(parents=True)
        with p.open('xb') as f: torch.save(child, f)
    (W / 'readers_done.json').write_text(json.dumps(dict(sources=sources, revert_order=order)))
else:
    bundle = R / 'studies/v28_frozen/bundles/subj_b25/eval'
    arms = [('I_P4', '0', '0-7,32-39', None, R / 'runtime_data/v27_pcache_gpu3'), ('G24_P4', '1', '8-15,40-47', W / 'readers/G24', R / 'runtime_data/v27_pcache_gpu2'),
            ('G20_P4', '2', '28-31,60-63', W / 'readers/G20', Path('/home/shensicheng/qyd/pans/work/pans_kv/pcache_a2_gpu2')),
            ('G16_P4', '3', '48-59', W / 'readers/G16', Path('/home/shensicheng/qyd/pans/work/pans_kv/pcache_a2_gpu3'))]
    def one(a):
        name, g, cores, ad, pc = a; out = W / 'eval' / f'test_{name}'; out.parent.mkdir(parents=True, exist_ok=True)
        sh(f'eval_{name}', [PY, 'scripts/run_evidence_v26.py', *chain(out, 'identity' if ad is None else 'shared', bundle, 1003, adapters=ad, pcache=pc)], g, cores)
    ts = [threading.Thread(target=one, args=(a,)) for a in arms]; [t.start() for t in ts]; [t.join() for t in ts]
    (W / 'eval_done.json').write_text('{}')
