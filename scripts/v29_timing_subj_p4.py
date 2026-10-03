"""v29 SUBJ/25% P4 parallel setting: same-block SSD-cold timing on the main pcache system, arms I_P8 / G16_P8 (frozen v28 R0) /
I_P4 / G16_P4 (studies/v29/subj_p4/readers/G16), 6 blocks x 16 queries, Williams orders. Usage: v29_timing_subj_p4.py [--dry-run]"""
from pathlib import Path
import json, runpy, sys
R = Path('/home/panzihang/src/prism_layerkv')
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py')); PY = ns['PY']
C16 = R / 'runtime_data/v29_pcache_gpu2_ssd'
AD = {'G16_P8': R / 'studies/v28_frozen/readers/subj_b25', 'G16_P4': R / 'studies/v29/subj_p4/readers/G16'}
WILLIAMS = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1], [0, 3, 1, 2], [1, 0, 2, 3]]
ARMS = ['I_P8', 'G16_P8', 'I_P4', 'G16_P4']
def argv(out, cell, arm, cond, block, name):
    o = out / 'runs' / name; period = arm[-1]
    mode = ['--mode', 'identity' if arm.startswith('I') else 'shared', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl']
    if arm in AD: mode += ['--adapters', str(AD[arm]), '--allow-candidate', '--packed-ops']
    return [PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', '/data1/llm/Qwen/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'),
            '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(C16),
            '--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(C16), '--cold-log', f'{o}.cold.jsonl', '--', *mode, '--',
            '--task', 'subj', '--budget', '25', '--period', period, '--max-tokens', '1', '--warmup', '4', '--samples', '16', '--profile', 'none',
            '--bundle-dir', str(R / f'studies/v28_frozen/bundles/subj_b25/b{block:02d}'), '--output', str(o)]
plan = [dict(cell='subj_b25', cond='preserve', block=b, order=[ARMS[i] for i in WILLIAMS[b]]) for b in range(6)]
ns['execute']('timing_subj_p4', plan, argv, dry='--dry-run' in sys.argv, meta=dict(orders=WILLIAMS))
