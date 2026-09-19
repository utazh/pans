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
    p.add_argument("--backend",choices=["reference","flash2","vllm2"],default="vllm2")
    p.add_argument("--adapters",type=Path)
    p.add_argument("--allow-diagnostic",action="store_true")
    p.add_argument("--trace-dir",type=Path)
    p.add_argument("--stats",type=Path)
    p.add_argument("--logits",type=Path)
    p.add_argument("--selection-plan",type=Path)
    p.add_argument("--max-query-samples",type=int,default=8)
    p.add_argument("round4_args",nargs=argparse.REMAINDER)
    a=p.parse_args()
    argv=a.round4_args
    if argv and argv[0]=="--":argv=argv[1:]
    if not argv:raise ValueError("Pass run_round4.py arguments after --")
    checks={'scripts/run_round4.py': 'd9b973eb528c26e713f75b8620ec7cc9608b8dc8', 'src/contiguous_fuxian/flexgen_qwen_reprefill.py': 'd52f1f610b7e63d9b150f7bfdba2caaa2866101e', 'src/contiguous_fuxian/flexgen_pcache.py': 'b86364086588327f9b9a06e608968aa16b2ce20c'}
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
        SharedRuntime(mode=a.mode,backend=a.backend,adapter_dir=a.adapters,stats_path=a.stats,allow_diagnostic=a.allow_diagnostic).install(runner)
    if a.selection_plan:
        import torch
        from contiguous_fuxian.sharedkv.io import query_hash
        unfrozen_completion = runner.greedy_flexgen_completion
        def frozen_completion(*args, **kw):
            path=a.selection_plan/(query_hash(kw["query_token_ids"])+".pt")
            chosen=torch.load(path,map_location="cpu",weights_only=True)["selected_by_layer"]
            loader=kw["loader"]
            from contiguous_fuxian.sharedkv.selection import fixed_selection
            with fixed_selection(loader,chosen):
                return unfrozen_completion(*args,**kw)
        runner.greedy_flexgen_completion=frozen_completion
    if a.logits:
        import torch
        from contiguous_fuxian.sharedkv.io import query_hash
        a.logits.mkdir(parents=True,exist_ok=True)
        original_completion = runner.greedy_flexgen_completion
        def capture(*args, **kw):
            output = kw.get("first_token_logits_out")
            if output is None:
                output = []
                kw["first_token_logits_out"] = output
            result = original_completion(*args, **kw)
            torch.save({"logits":output[-1], "completion":result[0], "query_tokens":len(kw["query_token_ids"]), "query_hash":query_hash(kw["query_token_ids"]),
                        "selected_by_layer":kw["loader"]._selected_by_layer},
                       a.logits / (query_hash(kw["query_token_ids"])+".pt"))
            return result
        runner.greedy_flexgen_completion = capture
    sys.argv=[str(ROOT/"scripts/run_round4.py"),*argv]
    runpy.run_path(str(ROOT/"scripts/run_round4.py"),run_name="__main__")


if __name__=="__main__":main()
