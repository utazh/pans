#!/usr/bin/env python3
"""v23 options -- existing run_sharedkv_v21 options -- existing round4 options."""
from pathlib import Path
import argparse,runpy,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def value(args,flag,default=None):
    if flag not in args:return default
    i=args.index(flag)
    if i+1>=len(args):raise ValueError(f'Missing {flag} value')
    return args[i+1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir');p.add_argument('--start',type=int);p.add_argument('--end',type=int)
    p.add_argument('--max-suffix',type=int,default=4)
    p.add_argument('--cold',choices=['none','app-cold','page-hot','ssd-cold'],default='none')
    p.add_argument('--owned-cache-root');p.add_argument('--cold-log')
    p.add_argument('--pcache-root',help='Explicit isolated physical Pcache directory')
    p.add_argument('--read-advice',choices=['native','preserve'],default='native')
    p.add_argument('v21_args',nargs=argparse.REMAINDER)
    a=p.parse_args();argv=a.v21_args
    if argv and argv[0]=='--':argv=argv[1:]
    if '--' not in argv:raise ValueError('Pass v21 options, then --, then round4 options')
    sep=argv.index('--');vargs,rargs=argv[:sep],argv[sep+1:]
    mode=value(vargs,'--mode');task=value(rargs,'--task')
    if not task or mode not in ('identity','shared','original'):raise ValueError('Explicit --task and --mode required')
    if a.capture_dir and (a.cold!='none' or mode=='original'):
        raise ValueError('Capture requires identity/shared, separate from cold/performance runs')
    if a.capture_dir:
        if a.start is None or a.end is None or '--diagnostic' not in vargs:
            raise ValueError('Capture requires --start --end and v21 --diagnostic')
        if int(value(rargs,'--warmup',32))!=0:raise ValueError('Capture warmup=0 prevents duplicate trace files')
        if int(value(rargs,'--max-tokens',1))!=1:raise ValueError('Use max_tokens=1; label scoring supplies forced tails')
        if '--trace-dir' in vargs or '--replay-dir' in vargs:raise ValueError('Use v23 capture and v21 --selection-plan, not sampled traces')
        if Path(a.capture_dir).exists():raise FileExistsError(a.capture_dir)
    if a.cold!='none':
        if a.cold=='page-hot' and a.read_advice!='preserve':raise ValueError('page-hot requires --read-advice preserve; native code evicts pages on each read')
        if not a.owned_cache_root or not a.cold_log:raise ValueError('Cold requires dedicated --owned-cache-root and --cold-log')
        warmup=int(value(rargs,'--warmup',32));samples=int(value(rargs,'--samples',1000000))
        if warmup<1 or samples<warmup:raise ValueError('Cold requires 1 <= explicit warmup <= samples')
        if any(x in vargs for x in ('--trace-dir','--replay-dir','--logits','--shadow-load','--selection-plan')):
            raise ValueError('Formal cold path retains ONLINE selection and excludes trace/logits/shadow/replay')
    from contiguous_fuxian.sharedkv_v23.guard import check
    check(ROOT)
    from contiguous_fuxian.sharedkv_v21.runtime import Runtime
    old_install=Runtime.install
    import contiguous_fuxian.flexgen_pcache as pcache_module
    old_loader=pcache_module.FlexGenPcacheStore.new_layer_loader
    import contiguous_fuxian.flexgen_qwen_reprefill as runner
    old_main=runner.main
    if a.pcache_root:
        cache_root=Path(a.pcache_root).resolve(strict=True)
        if a.cold!='none' and cache_root!=Path(a.owned_cache_root).resolve(strict=True):
            raise ValueError('Actual Pcache root must equal owned cold root')
        def cache_main(*args,**kwargs):
            i=sys.argv.index('--flexgen-kv-dir');sys.argv[i+1]=str(cache_root)
            return old_main(*args,**kwargs)
        runner.main=cache_main
    def install(self,runner):
        old_install(self,runner)
        if a.capture_dir:
            from contiguous_fuxian.sharedkv_v23.capture import GroupCollector
            GroupCollector(a.capture_dir,a.start,a.end,a.max_suffix).install(self,runner)
        if a.cold!='none':
            from contiguous_fuxian.sharedkv_v23.cold import ColdController
            ColdController(a.cold,a.owned_cache_root,a.cold_log,warmup,read_advice=a.read_advice).install(runner)
    Runtime.install=install
    if a.cold!='none':
        from contiguous_fuxian.sharedkv_v23.cold import install_owner_hook,ColdController
        install_owner_hook()
        if mode=='original':
            import contiguous_fuxian.flexgen_qwen_reprefill as runner
            ColdController(a.cold,a.owned_cache_root,a.cold_log,warmup,read_advice=a.read_advice).install(runner)
    sys.argv=[str(ROOT/'scripts/run_sharedkv_v21.py'),*argv]
    try:
        runpy.run_path(str(ROOT/'scripts/run_sharedkv_v21.py'),run_name='__main__')
    finally:
        Runtime.install=old_install
        pcache_module.FlexGenPcacheStore.new_layer_loader=old_loader
        runner.main=old_main

if __name__=='__main__':main()
