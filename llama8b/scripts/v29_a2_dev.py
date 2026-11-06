"""v29 A2 dev-set evaluation of the pilot R0 checkpoints (rule rev 2); each on the same GPU/pcache
as the cell's A0 identity dev run. Usage: v29_a2_dev.py"""
from pathlib import Path
import json, os, subprocess, threading, time
R = Path('/home/shensicheng/qyd/pans/src'); PY = '/home/shensicheng/qyd/pans/venv/bin/python'
OUT = R / 'studies/v29/a2_dev'; OUT.mkdir(parents=True, exist_ok=False); (OUT / 'runs').mkdir(); (OUT / 'logs').mkdir()
a0 = json.loads((R / 'studies/v29/a0/plan.json').read_text())
where = {j['task']: g for g, js in a0['jobs'].items() for j in js}
jobs = {}
for cell in ('sst2_b5', 'rte_b5'):
    task, b = cell.split('_')[0], int(cell.split('_b')[1]); g = where[task]
    for arm in ('fit64', 'fit512'):
        name = f'dev_A2{arm}_{cell}'; o = OUT / 'runs' / name
        jobs.setdefault(g, []).append(dict(name=name, argv=[
            PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'),
            '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', a0['queues'][g]['pcache'], '--',
            '--mode', 'shared', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl',
            '--adapters', str(R / f'studies/v29/readers/{cell}_A2{arm}'), '--allow-candidate', '--packed-ops', '--',
            '--task', task, '--budget', str(b), '--period', '8', '--max-tokens', '1', '--warmup', '0', '--samples', '500',
            '--profile', 'none', '--bundle-dir', str(R / 'data/v29/p0/dev'), '--output', str(o)]))
(OUT / 'plan.json').write_text(json.dumps(jobs, indent=1))
def run(g):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24')
    for j in jobs[g]:
        t0 = time.time()
        with (OUT / 'logs' / (j['name'] + '.log')).open('x') as f:
            rc = subprocess.run(['taskset', '-c', a0['queues'][g]['cores']] + j['argv'], cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        s = dict(name=j['name'], gpu=g, returncode=rc, seconds=round(time.time() - t0, 1))
        with (OUT / 'status.jsonl').open('a') as f: f.write(json.dumps(s) + '\n')
        print(json.dumps(s), flush=True)
ts = [threading.Thread(target=run, args=(g,)) for g in jobs]; [t.start() for t in ts]; [t.join() for t in ts]
(OUT / 'done.json').write_text(json.dumps(dict(finished=time.time()))); print('A2_DEV_DONE', flush=True)
