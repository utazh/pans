"""v29 P1-c: DIRECT same-block timing of our selected config against the system baselines (no cross-stage
normalization). Cells: 4 tasks x 25%. Arms per block: I, SEL (dev-selected shared config; for subj_b25 whose
selection is I the shared reference arm is G16, labelled G16), CKV, IMP. preserve (SSD-cold) + warm, 6 blocks x 16
queries. 4 arms: 6 fixed orders (4 Williams-design sequences + 2 mirrored), each arm appears in every position.
Same protocol/cores/gate/flags as rule rev 3. Usage: v29_timing_p1c.py [--dry-run]"""
from pathlib import Path
import json, runpy, sys
R = Path('/home/shensicheng/qyd/pans/src')
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py'))
final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
man = json.loads((R / 'studies/v29/readers/manifest.json').read_text())['cells']
C16 = R / 'runtime_data/v29_pcache_gpu2_ssd'; C64 = R / 'runtime_data/v29_pcache_impress_c64_ssd'
REORDER = R / 'configs/v29_impress_reorder_p0_fit4.json'; PY = ns['PY']
CELLS = ['sst2_b25', 'subj_b25', 'trec_b25', 'rte_b25']
def shared_arm(cell):
    sel = final[cell]['config']
    return ('G16', man[cell]['G16']['adapter_dir']) if sel == 'I' else (sel, final[cell]['adapter_dir'])
WILLIAMS = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1], [0, 3, 1, 2], [1, 0, 2, 3]]

def argv(out, cell, arm, cond, block, name):
    task, b = cell.split('_')[0], int(cell.split('_b')[1]); o = out / 'runs' / name
    kv = C64 if arm == 'IMP' else C16
    cold = ['--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(kv), '--cold-log', f'{o}.cold.jsonl'] if cond == 'preserve' else []
    tail = ['--task', task, '--budget', str(b), '--max-tokens', '1', '--warmup', '4', '--samples', '16',
            '--bundle-dir', str(R / f'studies/v28_frozen/bundles/{cell}/b{block:02d}'), '--output', str(o)]
    if arm in ('CKV', 'IMP'):
        method = ['--method', 'contigkv'] if arm == 'CKV' else ['--method', 'impress', '--impress-reorder', str(REORDER)]
        return [PY, str(R / 'scripts/v29_baseline_entry.py'), '--admission', 'original', '--audit-jsonl', f'{o}.v26.jsonl', '--',
                '--model-path', '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'), '--',
                '--pcache-root', str(kv), *cold, '--', '--mode', 'original', '--', *method, *tail]
    mode = ['--mode', 'identity' if arm == 'I' else 'shared', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl']
    if arm != 'I': mode += ['--adapters', shared_arm(cell)[1], '--allow-candidate', '--packed-ops']
    return [PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'),
            '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(kv), *cold, '--', *mode, '--',
            '--period', '8', '--profile', 'none', *tail]

plan = []
for cond in ('preserve', 'warm'):
    for cell in CELLS:
        arms = ['I', shared_arm(cell)[0], 'CKV', 'IMP']
        for block in range(6):
            plan.append(dict(cell=cell, cond=cond, block=block, order=[arms[i] for i in WILLIAMS[block]]))
ns['execute']('timing_p1c', plan, argv, dry='--dry-run' in sys.argv,
              meta=dict(shared_arms={c: shared_arm(c)[0] for c in CELLS}, orders=WILLIAMS))
