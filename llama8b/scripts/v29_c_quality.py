"""v29 stage C test-set quality: ContiguousKV and IMPRESS reproductions, 16 cells, same prefix/bundles/scoring
as our system (v28 frozen test bundles, warmup 0). 4 GPUs, each with its own Pcache copies. Cells are assigned
greedily by estimated cost so queues finish together."""
from pathlib import Path
import json, os, subprocess, threading, time
R = Path('/home/shensicheng/qyd/pans/src'); PY = '/home/shensicheng/qyd/pans/venv/bin/python'
OUT = R / 'studies/v29/c_quality'; OUT.mkdir(parents=True, exist_ok=False); (OUT / 'runs').mkdir(); (OUT / 'logs').mkdir()
REORDER = R / 'configs/v29_impress_reorder_p0_fit4.json'
N = {'sst2': 872, 'subj': 1003, 'trec': 500, 'rte': 277}
GPUS = {'0': dict(cores='0-7,32-39', c16=R / 'runtime_data/v27_pcache_gpu3', c64='/home/shensicheng/qyd/pans/work/pans_kv/pcache_impress_q0'),
        '1': dict(cores='8-15,40-47', c16=R / 'runtime_data/v27_pcache_gpu2', c64='/home/shensicheng/qyd/pans/work/pans_kv/pcache_impress_q1'),
        '2': dict(cores='28-31,60-63', c16='/home/shensicheng/qyd/pans/work/pans_kv/pcache_a2_gpu2', c64='/home/shensicheng/qyd/pans/work/pans_kv/pcache_impress_q2'),
        '3': dict(cores='48-59', c16='/home/shensicheng/qyd/pans/work/pans_kv/pcache_a2_gpu3', c64='/home/shensicheng/qyd/pans/work/pans_kv/pcache_impress_q3')}
cost_per_req = {('contigkv', 5): .5, ('contigkv', 10): .7, ('contigkv', 25): 1.3, ('contigkv', 50): 2.5,
                ('impress', 5): 2.5, ('impress', 10): 4, ('impress', 25): 9, ('impress', 50): 16}
jobs = sorted([(m, t, b) for m in ('impress', 'contigkv') for t in N for b in (5, 10, 25, 50)],
              key=lambda j: -cost_per_req[(j[0], j[2])] * N[j[1]])
load = {g: 0.0 for g in GPUS}; queue = {g: [] for g in GPUS}
for m, t, b in jobs:
    g = min(load, key=load.get); load[g] += cost_per_req[(m, b)] * N[t]
    name = f'{m}_{t}_b{b}'; o = OUT / 'runs' / name; kv = GPUS[g]['c64' if m == 'impress' else 'c16']
    extra = ['--impress-reorder', str(REORDER)] if m == 'impress' else []
    queue[g].append(dict(name=name, argv=[
        PY, str(R / 'scripts/v29_baseline_entry.py'), '--admission', 'original', '--audit-jsonl', f'{o}.v26.jsonl', '--',
        '--model-path', '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'), '--',
        '--pcache-root', str(kv), '--', '--mode', 'original', '--', '--method', m, *extra, '--task', t, '--budget', str(b),
        '--warmup', '0', '--samples', str(N[t]), '--bundle-dir', str(R / f'studies/v28_frozen/bundles/{t}_b{b}/eval'),
        '--output', str(o)]))
(OUT / 'plan.json').write_text(json.dumps(dict(estimated_hours={g: round(v / 3600, 1) for g, v in load.items()}, queues=queue), indent=1))
print({g: round(v / 3600, 1) for g, v in load.items()}, flush=True)
def run(g):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24')
    for j in queue[g]:
        t0 = time.time()
        with (OUT / 'logs' / (j['name'] + '.log')).open('x') as f:
            rc = subprocess.run(['taskset', '-c', GPUS[g]['cores']] + j['argv'], cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        s = dict(name=j['name'], gpu=g, returncode=rc, seconds=round(time.time() - t0, 1))
        with (OUT / 'status.jsonl').open('a') as f: f.write(json.dumps(s) + '\n')
        print(json.dumps(s), flush=True)
ts = [threading.Thread(target=run, args=(g,)) for g in GPUS]; [t.start() for t in ts]; [t.join() for t in ts]
(OUT / 'done.json').write_text(json.dumps(dict(finished=time.time()))); print('C_QUALITY_DONE', flush=True)
