"""Freeze final per-cell configuration after A1+A2 (rule rev 2), then run test-set quality for
I + selected config per cell. Pilot A2 candidates are judged by the same rule. 4 GPUs, one task
per GPU; I and selected of a cell share GPU/pcache. Usage: v29_freeze_final.py"""
from pathlib import Path
import hashlib, json, math, os, subprocess, threading, time
R = Path('/home/panzihang/src/prism_layerkv'); PY = '/home/panzihang/venvs/vllm-stable/bin/python'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze = json.loads((R / 'audit/v29_rule_freeze.json').read_text())
assert sha(R / 'configs/v29_selection_rule.json') == freeze['sha256']
sel = json.loads((R / 'results/v29_a1/selection.json').read_text())['cells']
man = json.loads((R / 'studies/v29/readers/manifest.json').read_text())['cells']

def load(p):
    rows = {r['uid']: r for r in map(json.loads, (p / 'scored_records.jsonl').read_text().splitlines())}
    return rows, list(json.loads((p / 'summary.json').read_text())['tasks'].values())[0]['mean_physical_prefetch_kv_bytes']

final, a2 = {}, {}
for cell, s in sel.items():
    assert s['status'] == 'selected'
    task = cell.split('_')[0]; pick = s['selected']
    if cell in ('sst2_b5', 'rte_b5'):  # A2 pilot candidates through the same rule
        I, Ib = load(R / f'studies/v29/a0/runs/devI_{cell}'); cI = sum(r['correct'] for r in I.values())
        tol = math.ceil(0.01 * len(I))
        for arm in ('fit64', 'fit512'):
            rows, b = load(R / f'studies/v29/a2_dev/runs/dev_A2{arm}_{cell}')
            net = sum(rows[u]['correct'] for u in I) - cI
            a2[f'{cell}_A2{arm}'] = dict(net=net, eligible=net >= -tol, bytes_vs_I=b / Ib - 1)
            assert net < -tol  # observed: none eligible -> A1 selection stands
    if pick == 'I':
        final[cell] = dict(config='I', mode='identity', adapter_dir=None, checkpoint_sha256=None)
    else:
        m = man[cell][pick]; ck = Path(m['adapter_dir']) / f'{task}.pt'
        assert sha(ck) == m['sha256']
        final[cell] = dict(config=pick, mode='shared', adapter_dir=m['adapter_dir'], checkpoint_sha256=m['sha256'])
rec = dict(rule_sha256=freeze['sha256'], rule_revision=2, frozen_unix=time.time(),
           a2_pilot=dict(results=a2, switch_fired=False,
                         note='fit512 - fit64 = -16 (sst2_b5), -3 (rte_b5); switch requires >= +5 in both; no recalibration'),
           cells=final)
with (R / 'audit/v29_final_selection.json').open('x') as f: json.dump(rec, f, indent=1)
print('FROZEN', sha(R / 'audit/v29_final_selection.json'), flush=True)

OUT = R / 'studies/v29/test_quality'; OUT.mkdir(parents=True, exist_ok=False); (OUT / 'runs').mkdir(); (OUT / 'logs').mkdir()
N = {'sst2': 872, 'subj': 1003, 'trec': 500, 'rte': 277}
GPU = {'sst2': ('0', '0-7,32-39', R / 'runtime_data/v27_pcache_gpu3'), 'subj': ('1', '8-15,40-47', R / 'runtime_data/v27_pcache_gpu2'),
       'trec': ('2', '16-23,48-55', '/data1/pans/pans_kv/pcache_a2_gpu2'), 'rte': ('3', '24-31,56-63', '/data1/pans/pans_kv/pcache_a2_gpu3')}
jobs = {}
for cell, f in final.items():
    task, b = cell.split('_')[0], int(cell.split('_b')[1]); g, cores, pc = GPU[task]
    bundle = R / f'studies/v28_frozen/bundles/{cell}/eval'
    for arm in (['I'] if f['config'] == 'I' else ['I', f['config']]):
        name = f'test_{arm}_{cell}'; o = OUT / 'runs' / name
        mode = ['--mode', 'identity', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl']
        if arm != 'I': mode += ['--adapters', f['adapter_dir'], '--allow-candidate', '--packed-ops']; mode[1] = 'shared'
        jobs.setdefault(g, dict(cores=cores, jobs=[]))['jobs'].append(dict(name=name, argv=[
            PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', '/data1/llm/Qwen/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'),
            '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(pc), '--', *mode, '--',
            '--task', task, '--budget', str(b), '--period', '8', '--max-tokens', '1', '--warmup', '0', '--samples', str(N[task]),
            '--profile', 'none', '--bundle-dir', str(bundle), '--output', str(o)]))
(OUT / 'plan.json').write_text(json.dumps(dict(final_selection_sha256=sha(R / 'audit/v29_final_selection.json'), jobs=jobs), indent=1))
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
(OUT / 'done.json').write_text(json.dumps(dict(finished=time.time()))); print('TEST_QUALITY_DONE', flush=True)
