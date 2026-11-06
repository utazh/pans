"""v29 A1: dev-set quality of shared candidates G16/G20/G24 for all 16 cells (rule rev 1).
Each cell runs on the same GPU/pcache as its A0 identity dev run (paired comparison).
Usage: v29_a1_sequence.py [--smoke]"""
from pathlib import Path
import json, os, subprocess, sys, threading, time
R = Path('/home/panzihang/src/prism_layerkv'); PY = '/home/panzihang/venvs/vllm-stable/bin/python'
SMOKE = '--smoke' in sys.argv
OUT = R / ('studies/v29/a1_smoke' if SMOKE else 'studies/v29/a1'); OUT.mkdir(parents=True, exist_ok=False)
man = json.loads((R / 'studies/v29/readers/manifest.json').read_text())
freeze = json.loads((R / 'audit/v29_rule_freeze.json').read_text())
assert man['rule_sha256'] == freeze['sha256']
a0 = json.loads((R / 'studies/v29/a0/plan.json').read_text())
dev = R / 'data/v29/p0/dev'

def job(name, task, budget, adapter_dir, samples, pcache):
    o = OUT / 'runs' / name
    return dict(name=name, task=task, budget=budget, adapter_dir=adapter_dir, samples=samples, argv=[
        PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
        '--model-path', '/data1/llm/Qwen/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'),
        '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--',
        '--pcache-root', str(pcache), '--', '--mode', 'shared', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid',
        '--stats', f'{o}.extra.jsonl', '--adapters', adapter_dir, '--allow-candidate', '--packed-ops', '--',
        '--task', task, '--budget', str(budget), '--period', '8', '--max-tokens', '1', '--warmup', '0',
        '--samples', str(samples), '--profile', 'none', '--bundle-dir', str(dev), '--output', str(o)])

plan = {}
for g, q in a0['queues'].items():
    tasks = sorted({j['task'] for j in a0['jobs'][g] if j['name'].startswith('devI_')})
    js = []
    for t in tasks:
        for b in (5, 10, 25, 50):
            cell = f'{t}_b{b}'
            for cand in ('G16', 'G20', 'G24'):
                js.append(job(f'dev_{cand}_{cell}', t, b, man['cells'][cell][cand]['adapter_dir'], 500, q['pcache']))
    plan[g] = js
if SMOKE:
    plan = {'0': [dict(plan['0'][1], samples=8, argv=[x if x != '500' else '8' for x in plan['0'][1]['argv']])]}
queues = a0['queues']
(OUT / 'plan.json').write_text(json.dumps(dict(rule_sha256=freeze['sha256'], queues=queues, jobs=plan), indent=1))
(OUT / 'runs').mkdir(); (OUT / 'logs').mkdir()

def run(g):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24')
    for j in plan[g]:
        t0 = time.time()
        with (OUT / 'logs' / (j['name'] + '.log')).open('x') as f:
            rc = subprocess.run(['taskset', '-c', queues[g]['cores']] + j['argv'], cwd=R, env=env,
                                stdout=f, stderr=subprocess.STDOUT).returncode
        s = dict(name=j['name'], gpu=g, returncode=rc, seconds=round(time.time() - t0, 1))
        with (OUT / 'status.jsonl').open('a') as f:
            f.write(json.dumps(s) + '\n')
        print(json.dumps(s), flush=True)

ts = [threading.Thread(target=run, args=(g,)) for g in plan]
[t.start() for t in ts]; [t.join() for t in ts]
(OUT / 'done.json').write_text(json.dumps(dict(finished=time.time())))
print('A1_DONE', flush=True)
