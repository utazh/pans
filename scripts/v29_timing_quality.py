"""v29 QuALITY formal timing on the main (pcache) system, SSD-cold, 25%: arms I / G24 / G16 (frozen syn readers).
Each of the 12 articles is one prefix block running all of its real questions (warmup 4); the 6 orders of the 3 arms are
each used by 2 articles. Same protocol/cores/gate/flags as rule rev 3. Usage: v29_timing_quality.py [--dry-run]"""
from pathlib import Path
import itertools, json, runpy, sys
R = Path('/home/panzihang/src/prism_layerkv')
ns = runpy.run_path(str(R / 'scripts/v29_timing_lib.py')); PY = ns['PY']
SETS, SLOTS = ['f0', 'f1', 'f2'], ['sst2', 'subj', 'trec', 'rte']
def n_rows(p): return sum(1 for l in Path(p).read_text().splitlines() if l.strip())
def argv(out, cell, arm, cond, block, name):
    s, slot = cell.split('_'); o = out / 'runs' / name; pc = R / f'runtime_data/v29_quality_{s}_pcache'
    bundle = R / f'data/v29/quality/{s}/eval'
    mode = ['--mode', 'identity' if arm == 'I' else 'shared', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl']
    if arm != 'I': mode += ['--adapters', str(R / f'studies/v29/quality/{s}/readers/syn_{arm}'), '--allow-candidate', '--packed-ops']
    return [PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', '/data1/llm/Qwen/Qwen2.5-7B-Instruct', '--store-root', str(R / f'runtime_data/v29_quality_{s}_store'),
            '--selector-index-dir', str(R / f'assets/v29_quality_{s}_index'), '--', '--pcache-root', str(pc),
            '--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(pc), '--cold-log', f'{o}.cold.jsonl', '--',
            *mode, '--', '--task', slot, '--budget', '25', '--period', '8', '--max-tokens', '1', '--warmup', '4',
            '--samples', str(n_rows(bundle / f'{slot}.jsonl')), '--profile', 'none', '--bundle-dir', str(bundle), '--output', str(o)]
perms = list(itertools.permutations(['I', 'G24', 'G16']))
cells = [f'{s}_{slot}' for s in SETS for slot in SLOTS]
plan = [dict(cell=c, cond='preserve', block=0, order=list(perms[i % 6])) for i, c in enumerate(cells)]
ns['execute']('timing_quality', plan, argv, dry='--dry-run' in sys.argv, meta=dict(note='one block per article; 6 orders x 2 articles'))
