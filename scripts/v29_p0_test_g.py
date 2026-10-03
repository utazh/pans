"""v29 P0: test-set quality of G20/G24 in every cell where they were not the selected config.
Reporting only (same-G comparison with cross-layer baselines); never used for selection.
Same GPU/pcache per task as studies/v29/test_quality so runs pair with test_I_<cell>."""
from pathlib import Path
import json, os, subprocess, threading, time
R = Path('/home/panzihang/src/prism_layerkv'); PY = '/home/panzihang/venvs/vllm-stable/bin/python'
OUT = R / 'studies/v29/p0_test_g'; OUT.mkdir(parents=True, exist_ok=False); (OUT / 'runs').mkdir(); (OUT / 'logs').mkdir()
final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
man = json.loads((R / 'studies/v29/readers/manifest.json').read_text())['cells']
N = {'sst2': 872, 'subj': 1003, 'trec': 500, 'rte': 277}
GPU = {'sst2': ('0', '0-7,32-39', R / 'runtime_data/v27_pcache_gpu3'), 'subj': ('1', '8-15,40-47', R / 'runtime_data/v27_pcache_gpu2'),
       'trec': ('2', '28-31,60-63', '/data1/pans/pans_kv/pcache_a2_gpu2'), 'rte': ('3', '48-59', '/data1/pans/pans_kv/pcache_a2_gpu3')}
jobs = {}
for cell, f in final.items():
    task, b = cell.split('_')[0], int(cell.split('_b')[1]); g, cores, pc = GPU[task]
    for cand in ('G20', 'G24'):
        if f['config'] == cand: continue
        name = f'test_{cand}_{cell}'; o = OUT / 'runs' / name
        jobs.setdefault(g, dict(cores=cores, jobs=[]))['jobs'].append(dict(name=name, argv=[
            PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', '/data1/llm/Qwen/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'),
            '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(pc), '--',
            '--mode', 'shared', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl',
            '--adapters', man[cell][cand]['adapter_dir'], '--allow-candidate', '--packed-ops', '--',
            '--task', task, '--budget', str(b), '--period', '8', '--max-tokens', '1', '--warmup', '0', '--samples', str(N[task]),
            '--profile', 'none', '--bundle-dir', str(R / f'studies/v28_frozen/bundles/{cell}/eval'), '--output', str(o)]))
(OUT / 'plan.json').write_text(json.dumps(jobs, indent=1)); print({g: len(v['jobs']) for g, v in jobs.items()}, flush=True)
def run(g):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24')
    for j in jobs[g]['jobs']:
        t0 = time.time()
        with (OUT / 'logs' / (j['name'] + '.log')).open('x') as f:
            rc = subprocess.run(['taskset', '-c', jobs[g]['cores']] + j['argv'], cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        s = dict(name=j['name'], gpu=g, returncode=rc, seconds=round(time.time() - t0, 1))
        with (OUT / 'status.jsonl').open('a') as f: f.write(json.dumps(s) + '\n')
        print(json.dumps(s), flush=True)
ts = [threading.Thread(target=run, args=(g,)) for g in jobs]; [t.start() for t in ts]; [t.join() for t in ts]
(OUT / 'done.json').write_text(json.dumps(dict(finished=time.time()))); print('P0_DONE', flush=True)
