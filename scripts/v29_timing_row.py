"""v29 unified row-engine timing (SSD-cold only; GPT plan). Part A: 4 tasks x 25%, arms I_ROW / SEL_ROW / XKV_ROW (subj_b25
selected I -> duplicate arm merged; G16_ROW listed as the shared reference), all 6 orders of 3 arms over 6 blocks.
Part B: trec_b25 same-batch control with 5 arms I_PC, S_PC (pcache path, v23 cold on the pcache), I_ROW, S_ROW, XKV_ROW,
6 blocks with rotated orders (each arm in each position at least once). Row arms: scripts/v29_rowengine_entry.py with
--row-cold (certification on the files actually read). Usage: v29_timing_row.py [--dry-run]"""
from pathlib import Path
import itertools, json, runpy, sys
R = Path('/home/panzihang/src/prism_layerkv')
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py')); PY = ns['PY']
final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
man = json.loads((R / 'studies/v29/readers/manifest.json').read_text())['cells']
C16 = R / 'runtime_data/v29_pcache_gpu2_ssd'; ROWS = R / 'runtime_data/v29_rows_ssd'; XKV = R / 'runtime_data/v29_xkv_ssd'
CELLS = ['sst2_b25', 'subj_b25', 'trec_b25', 'rte_b25']
def shared(cell):
    sel = final[cell]['config']
    return ('G16', man[cell]['G16']['adapter_dir']) if sel == 'I' else (sel, final[cell]['adapter_dir'])
def argv(out, cell, arm, cond, block, name):
    task = cell.split('_')[0]; o = out / 'runs' / name
    tail = ['--task', task, '--budget', '25', '--period', '8', '--max-tokens', '1', '--warmup', '4', '--samples', '16', '--profile', 'none',
            '--bundle-dir', str(R / f'studies/v28_frozen/bundles/{cell}/b{block:02d}'), '--output', str(o)]
    is_shared = arm in ('S_PC', 'S_ROW', 'SEL_ROW', 'G16_ROW')
    mode = ['--mode', 'shared' if is_shared else 'identity', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl']
    if is_shared:
        ad = man[cell]['G16']['adapter_dir'] if arm in ('S_PC', 'S_ROW', 'G16_ROW') and cell == 'trec_b25' or arm == 'G16_ROW' else shared(cell)[1]
        mode += ['--adapters', ad, '--allow-candidate', '--packed-ops']
    chain = ['--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--', '--model-path', '/data1/llm/Qwen/Qwen2.5-7B-Instruct',
             '--store-root', str(R / 'runtime_data/prefix_store'), '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--',
             '--pcache-root', str(C16)]
    if arm in ('I_PC', 'S_PC'):
        cold = ['--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(C16), '--cold-log', f'{o}.cold.jsonl']
        return [PY, str(R / 'scripts/run_evidence_v26.py')] + chain + cold + ['--', *mode, '--', *tail]
    eng, dd = ('xkv', XKV / cell) if arm == 'XKV_ROW' else ('native', ROWS / task)
    return [PY, str(R / 'scripts/v29_rowengine_entry.py'), '--engine', eng, '--data-dir', str(dd), '--row-cold', '--row-log', f'{o}.row.json'] + \
           chain + ['--', *mode, '--', *tail]
plan = []
perm3 = list(itertools.permutations([0, 1, 2]))
for cell in CELLS:
    arms = ['I_ROW', 'G16_ROW' if final[cell]['config'] == 'I' else 'SEL_ROW', 'XKV_ROW']
    for b in range(6): plan.append(dict(cell=cell, cond='preserve', block=b, order=[arms[i] for i in perm3[b]]))
arms5 = ['I_PC', 'S_PC', 'I_ROW', 'S_ROW', 'XKV_ROW']
rot = [[(i + s) % 5 for i in range(5)] for s in range(5)] + [[4, 3, 2, 1, 0]]
for b in range(6): plan.append(dict(cell='trec_b25', cond='preserve_ctrl', block=b, order=[arms5[i] for i in rot[b]]))
def argv2(out, cell, arm, cond, block, name):  # 'preserve_ctrl' is SSD-cold as well (separate block family)
    return argv(out, cell, arm, 'preserve', block, name)
ns['execute']('timing_row', plan, argv2, dry='--dry-run' in sys.argv,
              meta=dict(engine='v29_rowengine_entry (pread runs, 2-thread prefetch at online window selection, no app cache)',
                        selected={c: shared(c)[0] for c in CELLS}, trec_ctrl_shared='G16 (= selected for trec_b25)'))
