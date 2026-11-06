#!/usr/bin/env python3
"""v29 baseline launcher: runs the unchanged scripts/run_evidence_v26.py chain but redirects its innermost
runpy call of scripts/run_round4.py to scripts/run_round4_methods.py (ContiguousKV / IMPRESS).
Use v21 --mode original. Arguments are exactly those of run_evidence_v26.py."""
import runpy, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TARGET = str(ROOT / 'scripts/run_round4.py'); METHODS = str(ROOT / 'scripts/run_round4_methods.py')
_orig = runpy.run_path
def run_path(path, *args, **kwargs):
    if str(path) == TARGET:
        sys.argv[0] = METHODS
        return _orig(METHODS, *args, **kwargs)
    return _orig(path, *args, **kwargs)
runpy.run_path = run_path
if '--mode' in sys.argv and sys.argv[sys.argv.index('--mode') + 1] != 'original':
    raise SystemExit('baseline launcher requires v21 --mode original')
sys.argv = [str(ROOT / 'scripts/run_evidence_v26.py')] + sys.argv[1:]
_orig(sys.argv[0], run_name='__main__')
