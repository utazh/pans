#!/usr/bin/env python3
"""v33 AS+H2O+LRU launcher (3090). Runs the unchanged scripts/run_evidence_v26.py chain and redirects only its innermost
runpy call of scripts/run_round4.py to scripts/run_round4_methods_as.py (the A100 baseline runner that adds the
AttentionStore + H2O + LRU method; contigkv/impress branches identical to scripts/run_round4_methods.py).
SSD-cold certification (sharedkv_v23/cold.py evict_application) only accepts the CKLFU ScoreUnit controller; AS uses the
LRU controller (LRUUnit). Exactly as the audited A100 patch (xfer/patch_cold_lru2.py) - but applied in THIS process only,
cold.py on disk is untouched: the check line is rewritten to also accept LRUUnit; the drain / per-item residency checks
that follow are unchanged. Arguments are exactly those of run_evidence_v26.py (use v21 --mode original)."""
import inspect, runpy, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
import contiguous_fuxian.sharedkv_v23.cold as cold
src = inspect.getsource(cold.evict_application)
old = ("    if type(controller).__name__!='ScoreUnit' or controller.cache_type!='CKLFU':\n"
       "        raise ValueError('Only audited CKLFU ScoreUnit reset supported')")
new = ("    kind=type(controller).__name__\n"
       "    if not ((kind=='ScoreUnit' and controller.cache_type=='CKLFU') or kind=='LRUUnit'):\n"
       "        raise ValueError('Only audited CKLFU ScoreUnit or LRUUnit reset supported')")
if src.count(new) == 1: pass                        # tree already carries the audited A100 patch (xfer/patch_cold_lru2.py)
else:
    assert src.count(old) == 1, 'cold.evict_application changed; refusing to patch'
    exec(compile(src.replace(old, new), cold.__file__, 'exec'), cold.__dict__)
TARGET = str(ROOT / 'scripts/run_round4.py'); METHODS = str(ROOT / 'scripts/run_round4_methods_as.py')
_orig = runpy.run_path
def run_path(path, *args, **kwargs):
    if str(path) == TARGET:
        sys.argv[0] = METHODS
        return _orig(METHODS, *args, **kwargs)
    return _orig(path, *args, **kwargs)
runpy.run_path = run_path
if '--mode' in sys.argv and sys.argv[sys.argv.index('--mode') + 1] != 'original':
    raise SystemExit('AS launcher requires v21 --mode original')
sys.argv = [str(ROOT / 'scripts/run_evidence_v26.py')] + sys.argv[1:]
_orig(sys.argv[0], run_name='__main__')
