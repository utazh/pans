"""v29 xKV dev-set selection (configs/v29_xkv_selection_rule.json, frozen). run: 4 cells x 2 ranks on the 500-query dev set
through the unified row engine (warm, no timing claims). select: same eligibility/selection rule as rule rev 3.
Usage: v29_xkv_dev.py run | select"""
from pathlib import Path
import hashlib, json, math, os, subprocess, sys, threading, time
R = Path('/home/panzihang/src/prism_layerkv'); PY = '/home/panzihang/venvs/vllm-stable/bin/python'
rule = R / 'configs/v29_xkv_selection_rule.json'
assert hashlib.sha256(rule.read_bytes()).hexdigest() == (R / 'audit/v29_xkv_selection_rule.sha256').read_text().split()[0]
CELLS = ['sst2_b25', 'subj_b25', 'trec_b25', 'rte_b25']
RANKS = {'XKV_256_384_BF16': R / 'runtime_data/v29_xkv_ssd', 'XKV_384_576_BF16': R / 'runtime_data/v29_xkv_ssd_r384'}
OUT = R / 'studies/v29/xkv_dev'
GPU = {'sst2': ('0', '0-7,32-39'), 'subj': ('1', '8-15,40-47'), 'trec': ('2', '28-31,60-63'), 'rte': ('3', '48-59')}
if sys.argv[1] == 'run':
    OUT.mkdir(parents=True, exist_ok=False); (OUT / 'runs').mkdir(); (OUT / 'logs').mkdir()
    def per_cell(cell):
        task = cell.split('_')[0]; g, cores = GPU[task]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24')
        for name, root in RANKS.items():
            o = OUT / 'runs' / f'{name}_{cell}'
            argv = [PY, str(R / 'scripts/v29_rowengine_entry.py'), '--engine', 'xkv', '--data-dir', str(root / cell), '--row-log', f'{o}.row.json',
                    '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--', '--model-path', '/data1/llm/Qwen/Qwen2.5-7B-Instruct',
                    '--store-root', str(R / 'runtime_data/prefix_store'), '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--',
                    '--pcache-root', str(R / 'runtime_data/v29_pcache_gpu2_ssd'), '--', '--mode', 'identity', '--backend', 'vllm2', '--metadata-cache',
                    '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl', '--', '--task', task, '--budget', '25', '--period', '8', '--max-tokens', '1',
                    '--warmup', '0', '--samples', '500', '--profile', 'none', '--bundle-dir', str(R / 'data/v29/p0/dev'), '--output', str(o)]
            t0 = time.time()
            with (OUT / 'logs' / f'{name}_{cell}.log').open('x') as f:
                rc = subprocess.run(['taskset', '-c', cores] + argv, cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            with (OUT / 'status.jsonl').open('a') as f: f.write(json.dumps(dict(name=f'{name}_{cell}', returncode=rc, seconds=round(time.time() - t0, 1))) + '\n')
    ts = [threading.Thread(target=per_cell, args=(c,)) for c in CELLS]; [t.start() for t in ts]; [t.join() for t in ts]
    (OUT / 'done.json').write_text('{}'); print('XKV_DEV_DONE', flush=True)
else:
    def load(p):
        return {r['uid']: r for r in map(json.loads, (p / 'scored_records.jsonl').read_text().splitlines())}
    sel, table = {}, []
    for cell in CELLS:
        I = load(R / f'studies/v29/a0/runs/devI_{cell}'); n = len(I); cI = sum(r['correct'] for r in I.values()); tol = math.ceil(0.01 * n)
        Ib = list(json.loads((R / f'studies/v29/a0/runs/devI_{cell}/summary.json').read_text())['tasks'].values())[0]['mean_physical_prefetch_kv_bytes']
        cands = {'I': dict(net=0, bytes=Ib, eligible=True, disagree=0)}
        for name in RANKS:
            p = OUT / 'runs' / f'{name}_{cell}'; X = load(p); assert set(X) == set(I)
            rb = json.loads(Path(f'{p}.row.json').read_text())['request_bytes']; assert len(rb) == n
            net = sum(X[u]['correct'] for u in I) - cI
            cands[name] = dict(net=net, bytes=sum(rb) / len(rb), eligible=net >= -tol, disagree=sum(X[u]['prediction'] != I[u]['prediction'] for u in I))
        order = ['I', 'XKV_256_384_BF16', 'XKV_384_576_BF16']
        elig = [k for k in order if cands[k]['eligible']]; bmin = min(cands[k]['bytes'] for k in elig)
        near = [k for k in elig if (cands[k]['bytes'] - bmin) / bmin < 0.005]
        pick = sorted(near, key=lambda k: (-cands[k]['net'], order.index(k)))[0]
        sel[cell] = dict(selected=pick, candidates=cands, tolerance=tol)
        for k, v in cands.items(): table.append(f"{cell:9s} {k:17s} net {v['net']:+4d} disagree {v['disagree']:3d} bytes {v['bytes'] / 1e6:6.1f} MB eligible {v['eligible']}")
        table.append(f'{cell:9s} -> {pick}')
    (R / 'results/v29_xkv_dev_selection.json').write_text(json.dumps(dict(rule_sha256=hashlib.sha256(rule.read_bytes()).hexdigest(), cells=sel), indent=1))
    print('\n'.join(table))
