"""v29 unified row-engine timing, part 2 (GPT 2026-10-02): 4 tasks x 25%, SSD-cold, arms I_ROW / SEL_ROW (subj: G16_ROW
reference) / XKV256_ROW / XKV384_ROW, 6 blocks with Williams orders (each arm in every position). Shows every candidate
point; selected points come from configs/v29_selection_rule.json (ours) and results/v29_xkv_dev_selection.json (xKV).
Usage: v29_timing_row2.py [--dry-run]"""
from pathlib import Path
import json, runpy, sys
R = Path('/home/panzihang/src/prism_layerkv')
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py')); PY = ns['PY']
final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
man = json.loads((R / 'studies/v29/readers/manifest.json').read_text())['cells']
C16 = R / 'runtime_data/v29_pcache_gpu2_ssd'; ROWS = R / 'runtime_data/v29_rows_ssd'
XKV = {'XKV256_ROW': R / 'runtime_data/v29_xkv_ssd', 'XKV384_ROW': R / 'runtime_data/v29_xkv_ssd_r384'}
CELLS = ['sst2_b25', 'subj_b25', 'trec_b25', 'rte_b25']
def shared(cell):
    sel = final[cell]['config']
    return ('G16_ROW', man[cell]['G16']['adapter_dir']) if sel == 'I' else ('SEL_ROW', final[cell]['adapter_dir'])
WILLIAMS = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1], [0, 3, 1, 2], [1, 0, 2, 3]]
def argv(out, cell, arm, cond, block, name):
    task = cell.split('_')[0]; o = out / 'runs' / name; is_shared = arm in ('SEL_ROW', 'G16_ROW')
    mode = ['--mode', 'shared' if is_shared else 'identity', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl']
    if is_shared: mode += ['--adapters', shared(cell)[1], '--allow-candidate', '--packed-ops']
    eng, dd = ('xkv', XKV[arm] / cell) if arm in XKV else ('native', ROWS / task)
    return [PY, str(R / 'scripts/v29_rowengine_entry.py'), '--engine', eng, '--data-dir', str(dd), '--row-cold', '--row-log', f'{o}.row.json',
            '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--', '--model-path', '/data1/llm/Qwen/Qwen2.5-7B-Instruct',
            '--store-root', str(R / 'runtime_data/prefix_store'), '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--',
            '--pcache-root', str(C16), '--', *mode, '--', '--task', task, '--budget', '25', '--period', '8', '--max-tokens', '1', '--warmup', '4',
            '--samples', '16', '--profile', 'none', '--bundle-dir', str(R / f'studies/v28_frozen/bundles/{cell}/b{block:02d}'), '--output', str(o)]
plan = []
for cell in CELLS:
    arms = ['I_ROW', shared(cell)[0], 'XKV256_ROW', 'XKV384_ROW']
    for b in range(6): plan.append(dict(cell=cell, cond='preserve', block=b, order=[arms[i] for i in WILLIAMS[b]]))
ns['execute']('timing_row2', plan, argv, dry='--dry-run' in sys.argv, meta=dict(orders=WILLIAMS))
