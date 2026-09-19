#!/usr/bin/env python3
"""Fixed P8 / uniform exact configurable budget; separate formal and profiler runs."""
import argparse,functools,json,os,sys,time,threading,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--max-tokens',type=int,default=1)
p.add_argument('--task',choices=['sst2','subj','trec','rte'],required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--samples',type=int,default=1000000)
p.add_argument('--warmup',type=int,default=32)
p.add_argument('--profile',choices=['none','viztracer','nsys','torch'],default='none')
p.add_argument('--period',type=int,choices=[1,2,4,8],default=8)
p.add_argument('--timing',choices=['deferred','sync'],default='deferred')
p.add_argument('--budget',type=int,choices=[5,10,25,50,100],default=25)
p.add_argument('--variant',choices=['optimized','baseline'],default='optimized')
p.add_argument('--model-path',default=os.environ.get('PANS_MODEL_PATH','Qwen/Qwen2.5-7B-Instruct'))
p.add_argument('--bundle-dir',type=Path,default=Path(os.environ.get('PANS_BUNDLE_DIR',ROOT/'data/paper_task_bundles_full_eval_strict')))
p.add_argument('--store-root',type=Path,default=Path(os.environ.get('PANS_STORE_ROOT',ROOT/'runtime_data/prefix_store')))
p.add_argument('--pcache-dir',type=Path,default=Path(os.environ.get('PANS_PCACHE_DIR',ROOT/'runtime_data/pcache_c16')))
p.add_argument('--selector-index-dir',type=Path,default=Path(os.environ.get('PANS_SELECTOR_INDEX_DIR',ROOT/'assets/selector_index_k4_g32')))
a=p.parse_args()
CODE_ROOT=ROOT if a.variant=='optimized' else ROOT/'audit/round4_baseline'
a.output=a.output.resolve()
a.output.relative_to(ROOT)
if a.output.exists():raise SystemExit(f'output exists: {a.output}')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(CODE_ROOT/'src'))
if a.profile=='viztracer':sys.path.insert(0,str(ROOT/'runtime_deps'))
import torch
import contiguous_fuxian.flexgen_qwen_reprefill as runner
if a.profile!='none':
    os.environ['PRISM_PROFILE_NVTX']='1'
    import contiguous_fuxian.flexgen_pcache as bridge
    from contiguous_fuxian.quantized_key_index import QuantizedKeyIndex
    def wrap(obj,name,label):
        original=getattr(obj,name)
        @functools.wraps(original)
        def traced(*args,**kwargs):
            torch.cuda.nvtx.range_push(label)
            try:return original(*args,**kwargs)
            finally:torch.cuda.nvtx.range_pop()
        setattr(obj,name,traced)
    for name in ['qwen_online_prefix_head_scores','prepare_impress_block_scores','select_promixed_gqa_blocks','configure_online_layer_selection','_cache_layout','_layer_causal_mask']:
        wrap(runner,name,name)
    for name in ['load_selector_keys','_load_selector_keys_sync','resolve','_resolve_impress','_resolve_entry','_update_loaded_layer_score','schedule','schedule_impress_next','schedule_impress_period','schedule_range','schedule_impress_missing','resolve_impress_speculation','prefetch_selector_keys','resolve_deferred_impress_compute']:
        wrap(bridge.FlexGenLayerLoader,name,name)
    wrap(QuantizedKeyIndex,'load_layer','index/H2D_and_dequantize')
    if a.variant == 'optimized':
        sys.path.insert(0, str(CODE_ROOT/'vendor/flexgen'))
        import my_pcache_fast
        wrap(my_pcache_fast, '_gather_host_parts', 'payload/host_gather')
        wrap(my_pcache_fast.PrefixKVLayer, '_batch_copy_chunk_sources', 'payload/prepare_and_pack')
        wrap(bridge, '_prefetch_source_chunk_counts', 'metadata/source_counts')
    original=runner.greedy_flexgen_completion
    state={'request':0,'hooks':False,'profile':None,'events':[]}
    @functools.wraps(original)
    def completion(*args,**kwargs):
        state['request']+=1
        rid=state['request']
        capture=rid>min(a.samples,a.warmup)
        if not state['hooks']:
            for lid,layer in enumerate(kwargs['model'].model.layers):
                layer.register_forward_pre_hook(lambda module,args,lid=lid:(torch.cuda.nvtx.range_push(f'decoder/layer_{lid:02d}'), None)[1])
                layer.register_forward_hook(lambda module,args,output:(torch.cuda.nvtx.range_pop(), None)[1])
            state['hooks']=True
        if capture and a.profile == 'viztracer' and not state.get('cleared'):
            state['profile'].clear()
            state['cleared'] = True
        if capture and state['profile'] is None:
            if a.profile=='viztracer':
                from viztracer import VizTracer
                tracer=VizTracer(tracer_entries=5000000,exclude_files=[str(Path(importlib.util.find_spec(name).origin).parent) for name in ['torch','transformers','numpy']],ignore_c_function=True)
                tracer.start();state['profile']=tracer
            elif a.profile=='nsys':
                torch.cuda.cudart().cudaProfilerStart();state['profile']=True
            else:
                profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
                profiler.start();state['profile']=profiler
        torch.cuda.nvtx.range_push(f'request/{rid}/'+('measured' if capture else 'warmup'))
        started=time.perf_counter_ns()
        try:return original(*args,**kwargs)
        finally:
            state['events'].append({'request':rid,'captured':capture,'start_ns':started,'end_ns':time.perf_counter_ns()})
            torch.cuda.nvtx.range_pop()
    runner.greedy_flexgen_completion=completion
if a.profile == 'viztracer':
    from viztracer import VizTracer
    tracer = VizTracer(tracer_entries=10000000,
                      exclude_files=[str(Path(importlib.util.find_spec(name).origin).parent) for name in ['torch','transformers','numpy']],
                      ignore_c_function=True, min_duration=5)
    tracer.start()
    state['profile'] = tracer
argv=[
'--model-path',a.model_path,
'--bundle-dir',str(a.bundle_dir),
'--store-root',str(a.store_root),
'--plan',str(ROOT/'configs/round2'/f'qwen25_k{a.budget:03d}_uniform.json'),
'--flexgen-root',str(CODE_ROOT/'vendor/flexgen'),'--flexgen-kv-dir',str(a.pcache_dir),
'--reuse-flexgen-kv','--tasks',a.task,'--store-tasks','sst2,subj,trec,rte',
'--output-dir',str(a.output),'--samples-per-task',str(a.samples),'--max-tokens',str(a.max_tokens),
'--accuracy-scoring','label_continuation_loglikelihood','--dtype','bfloat16','--device','cuda',
'--gpu-cache-mb','55','--cpu-cache-mb','131','--cache-type','CKLFU',
'--online-selection','--probe-query-heads','0,7,14,21','--selector-kv-head-ids','0,1,2,3',
'--impress-selection-block-size','16','--impress-selection-period-size',str(a.period),
'--promixed-gqa-selection','--promixed-reuse-mode','fixed','--promixed-coverage-fraction','0.5',
'--no-promixed-adaptive-coverage','--exact-layer-block-budget','--impress-async-prefetch',
'--impress-period-prefetch-size','1','--no-impress-rolling-period-prefetch','--no-impress-priority-prefetch',
'--impress-deferred-compute-timing' if a.timing=='deferred' else '--no-impress-deferred-compute-timing',
'--no-impress-value-ordered-prefetch','--no-defer-cache-score-updates',
'--impress-known-period-prefetch' if a.period>1 else '--no-impress-known-period-prefetch',
'--period-size','8','--subperiod-size','4','--expected-keep-ratio',str(a.budget/100),
'--warmup-passes','1' if a.warmup else '0','--warmup-samples-per-task',str(max(1,a.warmup)),
'--selector-index-dir',str(a.selector_index_dir)]
sys.argv=[runner.__file__]+argv
try:
    runner.main()
finally:
    if a.profile!='none' and state['profile'] is not None:
        if a.profile=='nsys':torch.cuda.cudart().cudaProfilerStop()
        elif a.profile=='viztracer':
            state['profile'].stop();state['profile'].save(str(a.output/'viztracer.json'))
        else:
            state['profile'].stop();state['profile'].export_chrome_trace(str(a.output/'torch_trace.json'))
        (a.output/'profile_requests.json').write_text(json.dumps(state['events'],indent=2))
if a.output.exists():
    (a.output/'command.json').write_text(json.dumps({'args':argv,'profile':a.profile,'variant':a.variant,'code_root':str(CODE_ROOT),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')},indent=2))

if a.variant == 'optimized':
    import my_pcache_fast
    (a.output/'engineering_metrics.json').write_text(json.dumps(my_pcache_fast._STAGING_POOL.metrics(),indent=2))
