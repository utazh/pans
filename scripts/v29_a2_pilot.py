"""v29 A2 data-size pilot (rule rev 2): sst2_b5 on GPU2, rte_b5 on GPU3, each serial:
for arm in fit64, fit512: teacher capture -> plan -> R4 train 512 steps (epoch sampling) -> frozen R0.
Dev-set evaluation is a separate step (same GPU/pcache as the A0 identity run)."""
from pathlib import Path
import json, os, subprocess, sys, threading, time, hashlib
R = Path('/home/shensicheng/qyd/pans/src'); PY = '/home/shensicheng/qyd/pans/venv/bin/python'
MODEL = '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct'
sys.path.insert(0, str(R / 'src'))
freeze = json.loads((R / 'audit/v29_rule_freeze.json').read_text())
rule = json.loads((R / 'configs/v29_selection_rule.json').read_text())
assert rule['revision'] == 2 and hashlib.sha256((R / 'configs/v29_selection_rule.json').read_bytes()).hexdigest() == freeze['sha256']
OUT = R / 'studies/v29/a2'; OUT.mkdir(parents=True, exist_ok=False); (OUT / 'logs').mkdir()
N = {'fit64': 64, 'fit512': 512}
CELLS = {'2': dict(cell='sst2_b5', task='sst2', budget=5, cores='16-23,48-55', pcache='/home/shensicheng/qyd/pans/work/pans_kv/pcache_a2_gpu2'),
         '3': dict(cell='rte_b5', task='rte', budget=5, cores='24-31,56-63', pcache='/home/shensicheng/qyd/pans/work/pans_kv/pcache_a2_gpu3')}

def sh(name, argv, gpu, cores):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24')
    t0 = time.time()
    with (OUT / 'logs' / f'{name}.log').open('x') as f:
        rc = subprocess.run(['taskset', '-c', cores] + argv, cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    s = dict(name=name, gpu=gpu, returncode=rc, seconds=round(time.time() - t0, 1), argv=argv)
    with (OUT / 'status.jsonl').open('a') as f: f.write(json.dumps(s) + '\n')
    print(json.dumps({k: v for k, v in s.items() if k != 'argv'}), flush=True)
    if rc: raise RuntimeError(f'{name} failed rc={rc}')

def pipeline(gpu):
    c = CELLS[gpu]; cell, task, b = c['cell'], c['task'], c['budget']
    for arm in ('fit64', 'fit512'):
        tag = f'v29_{cell}_{arm}'
        teacher = R / f'runtime_data/{tag}_teacher_fit'; res = R / f'results/{tag}_capture'
        sh(f'{tag}_capture', [PY, '-u', 'scripts/collect_teacher_v26.py', '--teacher-records', str(teacher), '--split', 'fit', '--',
            '--admission', 'original', '--audit-jsonl', f'{res}.v26.jsonl', '--', '--model-path', MODEL,
            '--store-root', str(R / 'runtime_data/prefix_store'), '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--',
            '--pcache-root', c['pcache'], '--', '--mode', 'identity', '--backend', 'vllm2', '--layout', 'hybrid', '--metadata-cache',
            '--stats', f'{res}.extra.jsonl', '--diagnostic', '--', '--task', task, '--bundle-dir', str(R / f'data/v29/p0/{arm}'),
            '--samples', str(N[arm]), '--warmup', '0', '--max-tokens', '1', '--budget', str(b), '--period', '8',
            '--profile', 'none', '--output', str(res)], gpu, c['cores'])
        # plan: v26 cell sources, design hashes = this arm's own fit hashes
        from contiguous_fuxian.sharedkv_v24.records import load, FIELDS, PLAN_FORMAT, assert_plan
        fit = load(teacher, task, 'fit'); val = load(R / f'runtime_data/v26_p0_{cell}_teacher_val', task, 'validation')
        assert len(fit) == N[arm], (tag, len(fit))
        base = json.loads((R / f'configs/v26_p0_{cell}_plan.json').read_text()); row = next(iter(fit.values()))
        for k in FIELDS: assert base[k] == row[k], k
        for item in list(fit.values()) + list(val.values()): assert_plan(item, base['sources'])
        plan = dict(base, design_query_hashes=sorted(fit), status='v29_a2_pilot_fixed_v26_sources', quality_certified=False,
                    v29=dict(rule_sha256=freeze['sha256'], arm=arm, parent_plan=f'configs/v26_p0_{cell}_plan.json'))
        plan_p = R / f'configs/{tag}_plan.json'
        with plan_p.open('x') as f: json.dump(plan, f, indent=2)
        ck = R / f'assets/v29_a2/{cell}_{arm}/{task}.pt'; ck.parent.mkdir(parents=True, exist_ok=False)
        sh(f'{tag}_train', [PY, '-u', 'scripts/train_v29_epoch.py', '--sample-log', str(R / f'audit/{tag}_samples.json'),
            '--native-blocks', '4', '--worklog', str(R / f'audit/{tag}_compute.json'), '--', '--model-path', MODEL,
            '--store-root', str(R / 'runtime_data/prefix_store'), '--task', task, '--teacher-fit', str(teacher),
            '--teacher-val', str(R / f'runtime_data/v26_p0_{cell}_teacher_val'), '--plan', str(plan_p),
            '--steps', '512', '--eval-every', '25', '--seed', '24', '--device', 'cuda', '--output', str(ck)], gpu, c['cores'])
        r0 = R / f'studies/v29/readers/{cell}_A2{arm}/{task}.pt'; r0.parent.mkdir(parents=True, exist_ok=False)
        sh(f'{tag}_export_r0', [PY, '-c', 'import sys;from contiguous_fuxian.sharedkv_v27.frozen_control import export_frozen_r0;'
            'export_frozen_r0(sys.argv[1],sys.argv[2])', str(ck), str(r0)], gpu, c['cores'])

errs = []
def guarded(g):
    try: pipeline(g)
    except Exception as e: errs.append(repr(e)); print('FAILED', g, repr(e), flush=True)
ts = [threading.Thread(target=guarded, args=(g,)) for g in CELLS]
[t.start() for t in ts]; [t.join() for t in ts]
(OUT / 'done.json').write_text(json.dumps(dict(finished=time.time(), errors=errs)))
print('A2_PILOT_DONE', errs, flush=True)
