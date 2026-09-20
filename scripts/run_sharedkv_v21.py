#!/usr/bin/env python3
"""Published-v1 runner, no whole-model/GPU performance certification implied."""
from pathlib import Path
import argparse,json,runpy,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['original','identity','shared'],required=True)
    p.add_argument('--backend',choices=['reference','flash2','vllm2'],default='vllm2')
    p.add_argument('--adapters');p.add_argument('--stats',type=Path);p.add_argument('--logits',type=Path)
    p.add_argument('--layout',choices=['split','hybrid'],default='split')
    p.add_argument('--schedule',choices=['window','demand'],default='window')
    p.add_argument('--packed-ops',action='store_true');p.add_argument('--metadata-cache',action='store_true')
    p.add_argument('--shadow-load',action='store_true')
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--allow-candidate',action='store_true')
    p.add_argument('--trace-dir');p.add_argument('--replay-dir')
    p.add_argument('--selection-plan',type=Path,help='Published v1 logits directory; selector compute remains charged')
    p.add_argument('--trace-layers');p.add_argument('--max-query-samples',type=int,default=8)
    p.add_argument('--nvtx',action='store_true');p.add_argument('round4_args',nargs=argparse.REMAINDER)
    a=p.parse_args();argv=a.round4_args
    if argv and argv[0]=='--':argv=argv[1:]
    if not argv:raise ValueError('Pass run_round4.py arguments after --')
    from contiguous_fuxian.sharedkv_v21.guard import check
    check(ROOT)
    if '--variant' in argv and argv[argv.index('--variant')+1]!='optimized':raise ValueError('Only optimized code tree')
    if '--profile' in argv and argv[argv.index('--profile')+1]!='none':raise ValueError('Use external profiling separately')
    if a.mode!='original' and a.backend=='reference' and not a.diagnostic:raise ValueError('Reference backend is diagnostic only')
    if a.selection_plan and (not a.diagnostic or a.replay_dir):raise ValueError('Fixed-plan requires diagnostic and is exclusive with replay-dir')
    if a.mode=='original' and (a.trace_dir or a.replay_dir or a.shadow_load or a.layout!='split' or a.schedule!='window' or a.packed_ops or a.metadata_cache):
        raise ValueError('Original cannot use v21 runtime options; identity is the execution control')
    for path in (a.stats,a.trace_dir,a.logits):
        if path and Path(path).exists():raise FileExistsError(path)
    if a.stats:
        a.stats.parent.mkdir(parents=True,exist_ok=True)
        a.stats.with_suffix(a.stats.suffix+'.context.json').write_text(json.dumps({
            'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            'fixed_ids':bool(a.selection_plan or a.replay_dir),'selector_compute_preserved':True,
            'formal_latency_eligible':not a.diagnostic and a.logits is None,
            'warning':'No automatic quality or speed certification. Fixed plans are only controlled diagnostics.'},indent=2))
    import contiguous_fuxian.flexgen_qwen_reprefill as runner
    if a.mode!='original':
        from contiguous_fuxian.sharedkv_v21.runtime import Runtime
        Runtime(mode=a.mode,backend=a.backend,adapter_dir=a.adapters,stats_path=a.stats,
            layout=a.layout,packed_ops=a.packed_ops,diagnostic=a.diagnostic,allow_candidate=a.allow_candidate,
            trace_dir=a.trace_dir,replay_dir=a.replay_dir,trace_layers=None if a.trace_layers is None else [int(x) for x in a.trace_layers.split(',')],
            max_samples=a.max_query_samples,annotate=a.nvtx,schedule=a.schedule,
            metadata_cache=a.metadata_cache,shadow_load=a.shadow_load,
            selection_plan_in_use=a.selection_plan is not None).install(runner)
    if a.selection_plan:
        from contiguous_fuxian.sharedkv.io import load_safe,query_hash
        from contiguous_fuxian.sharedkv.selection import fixed_selection
        unfrozen=runner.greedy_flexgen_completion
        def frozen(*args,**kw):
            digest=query_hash(kw['query_token_ids']);r=load_safe(a.selection_plan/f'{digest}.pt')
            if r.get('query_hash',digest)!=digest:raise ValueError('Fixed-plan query mismatch')
            with fixed_selection(kw['loader'],r['selected_by_layer']):return unfrozen(*args,**kw)
        runner.greedy_flexgen_completion=frozen
    if a.logits:
        import torch
        from contiguous_fuxian.sharedkv.io import query_hash
        a.logits.mkdir(parents=True,exist_ok=False);uncaptured=runner.greedy_flexgen_completion
        def capture(*args,**kw):
            out=kw.get('first_token_logits_out')
            if out is None:out=[];kw['first_token_logits_out']=out
            result=uncaptured(*args,**kw);digest=query_hash(kw['query_token_ids'])
            torch.save({'query_hash':digest,'logits':out[-1],'completion':result[0],
                'query_tokens':len(kw['query_token_ids']),'selected_by_layer':kw['loader']._selected_by_layer,
                'label_token_logprobs':kw.get('label_token_logprobs_out')},a.logits/f'{digest}.pt')
            return result
        runner.greedy_flexgen_completion=capture
    sys.argv=[str(ROOT/'scripts/run_round4.py'),*argv]
    runpy.run_path(str(ROOT/'scripts/run_round4.py'),run_name='__main__')
if __name__=='__main__':main()
