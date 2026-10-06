"""14B/32B 冷启动计时（与 7B stage-B 相同的块设计）：每格 arms = I、S3（≈G16 全部消费层）、开发集选中配置（若不是 I/S3）。
SSD 冷（preserve），6 块 × 16 条请求（studies/v28_frozen/bundles/<cell>/bNN，与 7B 相同的请求），3 臂用全部 6 种顺序、2 臂 AB/BA 交替。
执行器 = scripts/v29_timing_lib.py（PANS_TIMING_* 指定 GPU/计时核/磁盘）；pcache = runtime_data/v29m_pcache_timing（独立副本）。
用法（在模型 worktree 中）: v29m_timing.py <study_name> --tag s800 [--cells a,b] [--dry-run]"""
from pathlib import Path
import itertools, json, os, runpy, sys
R = Path(__file__).resolve().parents[1]
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py'))
PY, MODEL = os.environ['PANS_PY'], os.environ['PANS_MODEL_PATH']
args = sys.argv[1:]; study = args[0]
opt = lambda k, d=None: args[args.index(k) + 1] if k in args else d
W = R / 'studies' / ('v29m_' + opt('--tag', 's800'))
PCACHE = R / 'runtime_data/v29m_pcache_timing'
cells = (opt('--cells') or ','.join(sorted(p.stem for p in (W / 'selection').glob('*.json')))).split(',')
ARMS = {}
for cell in cells:
    sel = json.loads((W / 'selection' / f'{cell}.json').read_text())['selected']
    arms = {'I': None, 'S3': str(W / 'readers' / cell / 'S3')}
    if sel not in arms: arms[sel] = str(W / 'readers' / cell / sel)
    ARMS[cell] = arms
def argv(out, cell, arm, cond, block, name):
    task, b = cell.split('_b')[0], int(cell.split('_b')[1]); o = out / 'runs' / name
    mode = ['--mode', 'identity' if arm == 'I' else 'shared', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl']
    if arm != 'I': mode += ['--adapters', ARMS[cell][arm], '--allow-candidate', '--packed-ops']
    return [PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', MODEL, '--store-root', str(R / 'runtime_data/prefix_store'),
            '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(PCACHE),
            '--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(PCACHE), '--cold-log', f'{o}.cold.jsonl', '--', *mode, '--',
            '--task', task, '--budget', str(b), '--period', '8', '--max-tokens', '1', '--warmup', '4', '--samples', '16',
            '--profile', 'none', '--bundle-dir', str(R / f'studies/v28_frozen/bundles/{cell}/b{block:02d}'), '--output', str(o)]
plan = []
for cell in cells:
    names = list(ARMS[cell])
    orders = list(itertools.permutations(names)) if len(names) == 3 else [tuple(names), tuple(reversed(names))] * 3
    for block in range(6): plan.append(dict(cell=cell, cond='preserve', block=block, order=list(orders[block])))
ns['execute'](study, plan, argv, dry='--dry-run' in args, meta=dict(script='v29m_timing', arms=ARMS, model=MODEL, tag=str(W)))
