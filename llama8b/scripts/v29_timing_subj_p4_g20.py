"""v29 SUBJ/25% P4: paired SSD-cold timing I_P4 vs G20_P4 (studies/v29/subj_p4/readers/G20) on the main pcache system,
6 blocks x 16 queries, alternating order. Needed only to support the 'small loss under P4 still accelerates' statement.
Usage: v29_timing_subj_p4_g20.py [--dry-run]"""
from pathlib import Path
import runpy, sys
R = Path('/home/shensicheng/qyd/pans/src')
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py')); PY = ns['PY']
C16 = R / 'runtime_data/v29_pcache_gpu2_ssd'
def argv(out, cell, arm, cond, block, name):
    o = out / 'runs' / name
    mode = ['--mode', 'identity' if arm == 'I_P4' else 'shared', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl']
    if arm == 'G20_P4': mode += ['--adapters', str(R / 'studies/v29/subj_p4/readers/G20'), '--allow-candidate', '--packed-ops']
    return [PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'),
            '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(C16),
            '--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(C16), '--cold-log', f'{o}.cold.jsonl', '--', *mode, '--',
            '--task', 'subj', '--budget', '25', '--period', '4', '--max-tokens', '1', '--warmup', '4', '--samples', '16', '--profile', 'none',
            '--bundle-dir', str(R / f'studies/v28_frozen/bundles/subj_b25/b{block:02d}'), '--output', str(o)]
plan = [dict(cell='subj_b25', cond='preserve', block=b, order=['I_P4', 'G20_P4'] if b % 2 == 0 else ['G20_P4', 'I_P4']) for b in range(6)]
ns['execute']('timing_subj_p4_g20', plan, argv, dry='--dry-run' in sys.argv, meta=dict(note='2 arms, alternating order'))
