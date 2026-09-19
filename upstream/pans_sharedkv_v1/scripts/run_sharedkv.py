#!/usr/bin/env python3
"""Opt-in launcher over the repository's existing scripts/run_round4.py.

Example: python scripts/run_sharedkv.py --mode identity --backend flash2 --
         --task trec --budget 25 --period 8 --warmup 32 --output results/id
"""
from pathlib import Path
import argparse
import hashlib
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["original","collect","identity","shared"],required=True)
    p.add_argument("--backend",choices=["reference","flash2"],default="flash2")
    p.add_argument("--adapters",type=Path)
    p.add_argument("--trace-dir",type=Path)
    p.add_argument("--stats",type=Path)
    p.add_argument("--max-query-samples",type=int,default=8)
    p.add_argument("round4_args",nargs=argparse.REMAINDER)
    a=p.parse_args()
    argv=a.round4_args
    if argv and argv[0]=="--":argv=argv[1:]
    if not argv:raise ValueError("Pass run_round4.py arguments after --")
    checks={
      "scripts/run_round4.py":"4fd85b3e4fe3c9c88bbd4ec1619f2f38c24c1e54",
      "src/contiguous_fuxian/flexgen_qwen_reprefill.py":"d52f1f610b7e63d9b150f7bfdba2caaa2866101e",
      "src/contiguous_fuxian/flexgen_pcache.py":"b86364086588327f9b9a06e608968aa16b2ce20c",
    }
    for name,expected in checks.items():
        content=(ROOT/name).read_bytes()
        actual=hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()
        if actual!=expected:
            raise RuntimeError(f"Upstream file changed: {name}. Re-audit the integration rather than bypassing this guard.")
    if "--variant" in argv and argv[argv.index("--variant")+1]!="optimized":
        raise ValueError("Only the current optimized code tree is supported")
    if "--profile" in argv and argv[argv.index("--profile")+1]!="none":
        raise ValueError("Use unprofiled formal runs. Profile the new path externally in a separate process.")
    import contiguous_fuxian.flexgen_qwen_reprefill as runner
    if a.mode=="collect":
        if a.trace_dir is None:raise ValueError("--trace-dir is required")
        from contiguous_fuxian.sharedkv.collect import Collector
        Collector(a.trace_dir,a.max_query_samples).install(runner)
    elif a.mode in ("shared","identity"):
        from contiguous_fuxian.sharedkv.runtime import SharedRuntime
        if a.stats and a.stats.exists():raise FileExistsError(a.stats)
        SharedRuntime(mode=a.mode,backend=a.backend,adapter_dir=a.adapters,stats_path=a.stats).install(runner)
    sys.argv=[str(ROOT/"scripts/run_round4.py"),*argv]
    runpy.run_path(str(ROOT/"scripts/run_round4.py"),run_name="__main__")


if __name__=="__main__":main()
