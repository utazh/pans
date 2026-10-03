"""v29 P1-a/b mechanism ablations, 4 tasks x 25%, on the dev-selected shared graph (subj_b25: G16 reference graph).
  quality : test-set accuracy of DIRECT (same-graph direct sharing, no reader) and EXPLICIT (explicit reconstruction),
            same GPU/pcache per task as studies/v29/test_quality (pairs with test_I_/test_<SEL>_ runs).
  timing  : same-block arms I / READ (read-side reader, ours) / DIRECT / EXPLICIT; preserve + warm; 6 Williams orders.
Usage: v29_p1ab.py quality | timing [--dry-run]"""
from pathlib import Path
import json, os, runpy, subprocess, sys, threading, time
R = Path('/home/panzihang/src/prism_layerkv'); PY = '/home/panzihang/venvs/vllm-stable/bin/python'
final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
man = json.loads((R / 'studies/v29/readers/manifest.json').read_text())['cells']
CELLS = ['sst2_b25', 'subj_b25', 'trec_b25', 'rte_b25']
N = {'sst2': 872, 'subj': 1003, 'trec': 500, 'rte': 277}
def graph(cell):
    sel = final[cell]['config']
    return ('G16', man[cell]['G16']['adapter_dir']) if sel == 'I' else (sel, final[cell]['adapter_dir'])

def evidence(launcher, o, pcache, cold, mode, adapters, cell, samples, warmup, bundle):
    task = cell.split('_')[0]
    head = [PY, str(R / 'scripts' / launcher[0])] + launcher[1:]
    m = ['--mode', mode, '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl']
    if adapters: m += ['--adapters', adapters, '--allow-candidate', '--packed-ops']
    return head + ['--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
                   '--model-path', '/data1/llm/Qwen/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'),
                   '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(pcache), *cold, '--', *m, '--',
                   '--task', task, '--budget', '25', '--period', '8', '--max-tokens', '1', '--warmup', str(warmup), '--samples', str(samples),
                   '--profile', 'none', '--bundle-dir', str(bundle), '--output', str(o)]
LAUNCH = {'I': ['run_evidence_v26.py'], 'READ': ['run_evidence_v26.py'],
          'DIRECT': ['v29_exec_variant.py', '--variant', 'direct'], 'EXPLICIT': ['v29_exec_variant.py', '--variant', 'explicit']}

if sys.argv[1] == 'quality':
    OUT = R / 'studies/v29/p1ab_quality'; OUT.mkdir(parents=True, exist_ok=False); (OUT / 'runs').mkdir(); (OUT / 'logs').mkdir()
    GPU = {'sst2': ('0', '0-7,32-39', R / 'runtime_data/v27_pcache_gpu3'), 'subj': ('1', '8-15,40-47', R / 'runtime_data/v27_pcache_gpu2'),
           'trec': ('2', '28-31,60-63', '/data1/pans/pans_kv/pcache_a2_gpu2'), 'rte': ('3', '48-59', '/data1/pans/pans_kv/pcache_a2_gpu3')}
    jobs = {}
    for cell in CELLS:
        task = cell.split('_')[0]; g, cores, pc = GPU[task]; gname, ad = graph(cell)
        for arm in ('DIRECT', 'EXPLICIT'):
            name = f'test_{arm}_{gname}_{cell}'; o = OUT / 'runs' / name
            jobs.setdefault(g, dict(cores=cores, jobs=[]))['jobs'].append(dict(name=name, argv=evidence(
                LAUNCH[arm], o, pc, [], 'shared', ad, cell, N[task], 0, R / f'studies/v28_frozen/bundles/{cell}/eval')))
    (OUT / 'plan.json').write_text(json.dumps(jobs, indent=1))
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
    (OUT / 'done.json').write_text(json.dumps(dict(finished=time.time()))); print('P1AB_QUALITY_DONE', flush=True)
else:
    ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py'))
    C16 = R / 'runtime_data/v29_pcache_gpu2_ssd'
    WILLIAMS = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1], [0, 3, 1, 2], [1, 0, 2, 3]]
    ARMS = ['I', 'READ', 'DIRECT', 'EXPLICIT']
    def argv(out, cell, arm, cond, block, name):
        o = out / 'runs' / name
        cold = ['--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(C16), '--cold-log', f'{o}.cold.jsonl'] if cond == 'preserve' else []
        return evidence(LAUNCH[arm], o, C16, cold, 'identity' if arm == 'I' else 'shared', None if arm == 'I' else graph(cell)[1],
                        cell, 16, 4, R / f'studies/v28_frozen/bundles/{cell}/b{block:02d}')
    plan = [dict(cell=c, cond=cond, block=b, order=[ARMS[i] for i in WILLIAMS[b]])
            for cond in ('preserve', 'warm') for c in CELLS for b in range(6)]
    ns['execute']('timing_p1ab', plan, argv, dry='--dry-run' in sys.argv, meta=dict(graphs={c: graph(c)[0] for c in CELLS}, orders=WILLIAMS))
