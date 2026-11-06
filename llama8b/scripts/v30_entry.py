#!/usr/bin/env python3
"""v30 launcher: runs the unchanged scripts/run_evidence_v26.py chain (v26 -> v25 -> v23 -> v21) and redirects only
its innermost runpy call of scripts/run_round4.py to scripts/v30_round4_ext.py. Arguments are exactly those of
run_evidence_v26.py; the round4 section may use the extra v30 options (see v30_round4_ext.py)."""
import runpy, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TARGET = str(ROOT / 'scripts/run_round4.py'); EXT = str(ROOT / 'scripts/v30_round4_ext.py')
_orig = runpy.run_path
def run_path(path, *args, **kwargs):
    if str(path) == TARGET:
        sys.argv[0] = EXT
        return _orig(EXT, *args, **kwargs)
    return _orig(path, *args, **kwargs)
runpy.run_path = run_path
sys.argv = [str(ROOT / 'scripts/run_evidence_v26.py')] + sys.argv[1:]
_orig(sys.argv[0], run_name='__main__')
