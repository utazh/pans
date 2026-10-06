"""A100 版 stage-B 计时（与 3090 v29_timing.py 相同的块设计）：每格 arms = I, G16, selected(若不是 I/G16)，
SSD 冷（preserve，16 格）与 warm（25/50%），每块 16 条请求 × 6 块；3 臂用全部 6 种顺序，2 臂 AB/BA 交替。
执行器 = scripts/v29_timing_lib.py（GPU/计时核/磁盘由 PANS_TIMING_* 环境变量指定）。
用法: v29a_timing.py <study_name> [--dry-run] [--cells a,b] [--conditions preserve,warm]"""
from pathlib import Path
import itertools, json, os, runpy, sys
R = Path(__file__).resolve().parents[1]
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py')); PY = os.environ.get('PANS_PY', ns['PY'])
MODEL = os.environ['PANS_MODEL_PATH']
args = sys.argv[1:]; study = args[0]
opt = lambda k: args[args.index(k) + 1].split(',') if k in args else None
final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
man = json.loads((R / 'studies/v29/readers/manifest.json').read_text())['cells']
remap = lambda p: str(p).replace('/home/panzihang/src/prism_layerkv', str(R))
PCACHE = R / 'runtime_data/v29_pcache_gpu2_ssd'
def arms_of(cell):
    sel = final[cell]['config']; arms = {'I': None, 'G16': remap(man[cell]['G16']['adapter_dir'])}
    if sel not in arms: arms[sel] = remap(final[cell]['adapter_dir'])
    return arms
ARMS = {}
def argv(out, cell, arm, cond, block, name):
    task, b = cell.split('_')[0], int(cell.split('_b')[1]); o = out / 'runs' / name
    cold = ['--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(PCACHE), '--cold-log', f'{o}.cold.jsonl'] if cond == 'preserve' else []
    mode = ['--mode', 'identity' if arm == 'I' else 'shared', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl']
    if arm != 'I': mode += ['--adapters', ARMS[cell][arm], '--allow-candidate', '--packed-ops']
    return [PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', MODEL, '--store-root', str(R / 'runtime_data/prefix_store'),
            '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(PCACHE), *cold, '--', *mode, '--',
            '--task', task, '--budget', str(b), '--period', '8', '--max-tokens', '1', '--warmup', '4', '--samples', '16',
            '--profile', 'none', '--bundle-dir', str(R / f'studies/v28_frozen/bundles/{cell}/b{block:02d}'), '--output', str(o)]
cells = opt('--cells') or list(final)
conds = opt('--conditions') or ['preserve', 'warm']
plan = []
for cond in conds:
    for cell in cells:
        if cond == 'warm' and int(cell.split('_b')[1]) not in (25, 50): continue
        ARMS[cell] = arms_of(cell); names = list(ARMS[cell])
        orders = list(itertools.permutations(names)) if len(names) == 3 else [tuple(names), tuple(reversed(names))] * 3
        for block in range(6): plan.append(dict(cell=cell, cond=cond, block=block, order=list(orders[block])))
ns['execute'](study, plan, argv, dry='--dry-run' in args, meta=dict(script='v29a_timing', arms=ARMS, model=MODEL))
