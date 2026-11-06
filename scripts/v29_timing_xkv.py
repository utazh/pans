"""v29 P1 xKV timing: same-block arms I (our no-sharing system), SEL (dev-selected shared config; subj_b25 selected I so
G16 is used as the shared reference and labelled G16), XKV (xKV-CLF/P8-SSD, BF16 rank (256,384), U rows read from SSD,
reconstruction + RoPE inside TTFT, rights resident). 4 tasks x 25%, preserve (SSD-cold) + warm, 6 blocks x 16 queries,
all 6 orders of 3 arms. Cold for XKV: the U files are evicted and certified resident==0 before each measured request
(scripts/v29_xkv_entry.py --xkv-cold), in addition to the unchanged v23 Pcache eviction. Usage: v29_timing_xkv.py [--dry-run]"""
from pathlib import Path
import itertools, json, runpy, sys
R = Path('/home/panzihang/src/prism_layerkv')
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py')); PY = ns['PY']
final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
man = json.loads((R / 'studies/v29/readers/manifest.json').read_text())['cells']
C16 = R / 'runtime_data/v29_pcache_gpu2_ssd'; XKV = R / 'runtime_data/v29_xkv_ssd'
CELLS = ['sst2_b25', 'subj_b25', 'trec_b25', 'rte_b25']
def shared_arm(cell):
    sel = final[cell]['config']
    return ('G16', man[cell]['G16']['adapter_dir']) if sel == 'I' else (sel, final[cell]['adapter_dir'])
def argv(out, cell, arm, cond, block, name):
    task = cell.split('_')[0]; o = out / 'runs' / name
    cold = ['--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(C16), '--cold-log', f'{o}.cold.jsonl'] if cond == 'preserve' else []
    mode = ['--mode', 'shared' if arm not in ('I', 'XKV') else 'identity', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid',
            '--stats', f'{o}.extra.jsonl']
    if arm not in ('I', 'XKV'): mode += ['--adapters', shared_arm(cell)[1], '--allow-candidate', '--packed-ops']
    head = [PY, str(R / 'scripts/run_evidence_v26.py')]
    if arm == 'XKV':
        head = [PY, str(R / 'scripts/v29_xkv_entry.py'), '--xkv-dir', str(XKV / cell), '--xkv-log', f'{o}.xkv.json'] + (['--xkv-cold'] if cond == 'preserve' else [])
    return head + ['--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
                   '--model-path', '/data1/llm/Qwen/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'),
                   '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(C16), *cold, '--', *mode, '--',
                   '--task', task, '--budget', '25', '--period', '8', '--max-tokens', '1', '--warmup', '4', '--samples', '16', '--profile', 'none',
                   '--bundle-dir', str(R / f'studies/v28_frozen/bundles/{cell}/b{block:02d}'), '--output', str(o)]
orders = list(itertools.permutations([0, 1, 2]))
plan = []
for cond in ('preserve', 'warm'):
    for cell in CELLS:
        arms = ['I', shared_arm(cell)[0], 'XKV']
        for b in range(6): plan.append(dict(cell=cell, cond=cond, block=b, order=[arms[i] for i in orders[b]]))
ns['execute']('timing_xkv', plan, argv, dry='--dry-run' in sys.argv,
              meta=dict(xkv='xKV-CLF/P8-SSD BF16 rank (256,384), v28 representation; quality identical to v28 bridge (32/32 exact on trec_b25)',
                        shared_arms={c: shared_arm(c)[0] for c in CELLS}))
