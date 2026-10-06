"""A100 版 P1-c：25% 预算、4 任务，同块计时 4 臂 = I、SEL（开发集选中；选中为 I 时用 S3/G16 作共享参照臂）、CKV、IMP。
SSD 冷（preserve）与 warm，6 块 × 16 条请求，Williams 顺序（与 3090 v29_timing_p1c.py 完全相同的块设计与基线参数）。
7B（ROOT=src）：SEL 来自 audit/v29_final_selection.json；14B/32B：来自 studies/v29m_<tag>/selection。
用法（在模型 worktree 中）: v29m_timing_sys.py <study> --model-tag q7|q14|q32 [--tag s800] [--dry-run]"""
from pathlib import Path
import json, os, runpy, sys
R = Path(__file__).resolve().parents[1]
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py'))
B = Path('/home/shensicheng/qyd/pans'); PY, MODEL = os.environ['PANS_PY'], os.environ['PANS_MODEL_PATH']
args = sys.argv[1:]; study = args[0]; opt = lambda k, d=None: args[args.index(k) + 1] if k in args else d
MT = opt('--model-tag'); CELLS = ['sst2_b25', 'subj_b25', 'trec_b25', 'rte_b25']
C64 = B / 'work' / MT / 'pcache_impress_c64_timing'; REORDER = B / 'work' / MT / 'impress_reorder_fit4.json'
if MT == 'q7':
    C16 = R / 'runtime_data/v29_pcache_gpu2_ssd'
    final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
    man = json.loads((R / 'studies/v29/readers/manifest.json').read_text())['cells']
    fix = lambda p: str(p).replace('/home/panzihang/src/prism_layerkv', str(R))
    def shared_arm(cell):
        sel = final[cell]['config']
        return ('G16', fix(man[cell]['G16']['adapter_dir'])) if sel == 'I' else (sel, fix(final[cell]['adapter_dir']))
else:
    C16 = R / 'runtime_data/v29m_pcache_timing'; W = R / 'studies' / ('v29m_' + opt('--tag', 's800'))
    def shared_arm(cell):
        sel = json.loads((W / 'selection' / f'{cell}.json').read_text())['selected']
        sel = 'S3' if sel == 'I' else sel
        return (sel, str(W / 'readers' / cell / sel))
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
                '--model-path', MODEL, '--store-root', str(R / 'runtime_data/prefix_store'), '--',
                '--pcache-root', str(kv), *cold, '--', '--mode', 'original', '--', *method, *tail]
    mode = ['--mode', 'identity' if arm == 'I' else 'shared', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl']
    if arm != 'I': mode += ['--adapters', shared_arm(cell)[1], '--allow-candidate', '--packed-ops']
    return [PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', MODEL, '--store-root', str(R / 'runtime_data/prefix_store'),
            '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(kv), *cold, '--', *mode, '--',
            '--period', '8', '--profile', 'none', *tail]
plan = []
for cond in ('preserve', 'warm'):
    for cell in CELLS:
        arms = ['I', shared_arm(cell)[0], 'CKV', 'IMP']
        for block in range(6): plan.append(dict(cell=cell, cond=cond, block=block, order=[arms[i] for i in WILLIAMS[block]]))
ns['execute'](study, plan, argv, dry='--dry-run' in args,
              meta=dict(script='v29m_timing_sys', model=MODEL, shared_arms={c: shared_arm(c) for c in CELLS}, orders=WILLIAMS))
