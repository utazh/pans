#!/usr/bin/env python3
"""External profiling wrapper: profiled TTFT is diagnostic only."""
import argparse,functools,json,os,runpy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
p=argparse.ArgumentParser()
p.add_argument('--profiler',choices=['viztracer','nsys'],required=True)
p.add_argument('--profile-output',type=Path,required=True)
p.add_argument('args',nargs=argparse.REMAINDER)
a=p.parse_args(); argv=a.args[1:] if a.args and a.args[0]=='--' else a.args
if a.profile_output.exists(): raise FileExistsError(a.profile_output)
a.profile_output.mkdir(parents=True)
warm=int(argv[argv.index('--warmup')+1]) if '--warmup' in argv else 32
import torch
import contiguous_fuxian.flexgen_qwen_reprefill as runner
import contiguous_fuxian.flexgen_pcache as bridge
from contiguous_fuxian.quantized_key_index import QuantizedKeyIndex
os.environ['PRISM_PROFILE_NVTX']='1'
def wrap(obj,name,label):
 orig=getattr(obj,name)
 @functools.wraps(orig)
 def call(*args,**kw):
  with torch.cuda.nvtx.range(label):return orig(*args,**kw)
 setattr(obj,name,call)
for name in ['qwen_online_prefix_head_scores','prepare_impress_block_scores','select_promixed_gqa_blocks','configure_online_layer_selection']:
 wrap(runner,name,name)
for name in ['load_selector_keys','_load_selector_keys_sync','resolve','_resolve_entry','schedule','prefetch_selector_keys']:
 wrap(bridge.FlexGenLayerLoader,name,name)
wrap(QuantizedKeyIndex,'load_layer','index/H2D_and_dequantize')
sys.path.insert(0,str(ROOT/'vendor/flexgen'))
import my_pcache_fast
wrap(my_pcache_fast,'_gather_host_parts','payload/host_gather')
wrap(my_pcache_fast.PrefixKVLayer,'_batch_copy_chunk_sources','payload/prepare_and_pack')
state={'count':0,'active':False,'events':[]}
tracer=None
if a.profiler=='viztracer':
 sys.path.insert(0,str(ROOT/'runtime_deps'))
 from viztracer import VizTracer
 tracer=VizTracer(tracer_entries=10000000,exclude_files=[str(Path(torch.__file__).parent),'/home/panzihang/venvs/vllm-stable/lib/python3.10/site-packages/transformers','/home/panzihang/venvs/vllm-stable/lib/python3.10/site-packages/numpy'],ignore_c_function=True,min_duration=5)
 tracer.start()
# Cold preparation touches tens of thousands of files outside TTFT.
# Keep those Python calls out of the VizTracer ring; actual prefills/labels remain traced.
if tracer is not None and '--cold' in argv:
 import contiguous_fuxian.sharedkv_v23.cold as cold_module
 old_evict=cold_module.evict_application;old_pages=cold_module.prepare_pages
 def profile_evict(*args,**kwargs):
  if state['active']:tracer.stop()
  return old_evict(*args,**kwargs)
 def profile_pages(*args,**kwargs):
  result=old_pages(*args,**kwargs)
  if state['active']:tracer.start()
  return result
 cold_module.evict_application=profile_evict
 cold_module.prepare_pages=profile_pages
original_main=runner.main
def main():
 greedy=runner.greedy_flexgen_completion
 decoder=runner.flexgen_sparse_decoder_logits
 def decode(*args,**kw):
  phase='prefill' if kw['cache'].get_seq_length(0)==0 else 'decode_or_label'
  with torch.cuda.nvtx.range('phase/'+phase):return decoder(*args,**kw)
 runner.flexgen_sparse_decoder_logits=decode
 def completion(*args,**kw):
  state['count']+=1;rid=state['count'];capture=rid>warm
  if capture and not state['active']:
   torch.cuda.synchronize()
   if a.profiler=='nsys':torch.cuda.cudart().cudaProfilerStart()
   else:
    state['profile_hook_before_capture']=repr(sys.getprofile())
    tracer.stop();tracer.clear();tracer.start()
   state['active']=True
  begin=time.perf_counter_ns()
  try:
   with torch.cuda.nvtx.range(f'request/{rid}/'+('measured' if capture else 'warmup')):
    return greedy(*args,**kw)
  finally:state['events'].append({'request':rid,'captured':capture,'start_ns':begin,'end_ns':time.perf_counter_ns()})
 runner.greedy_flexgen_completion=completion
 return original_main()
runner.main=main
try:
 sys.argv=[str(ROOT/'scripts/run_native_prefix_v25.py'),*argv]
 runpy.run_path(sys.argv[0],run_name='__main__')
finally:
 if a.profiler=='nsys' and state['active']:torch.cuda.cudart().cudaProfilerStop()
 if tracer is not None:
  tracer.stop();tracer.parse()
  if tracer.total_entries<100:raise RuntimeError('VizTracer captured too few events')
  tracer.save(str(a.profile_output/'viztracer.json'))
  (a.profile_output/'validated.json').write_text(json.dumps({'events':tracer.total_entries,'overflow':tracer.data['viztracer_metadata']['overflow'],'profile_hook_before_capture':state.get('profile_hook_before_capture')}))
 (a.profile_output/'profile_requests.json').write_text(json.dumps(state['events'],indent=2))
 (a.profile_output/'command.json').write_text(json.dumps({'args':argv,'profiler':a.profiler,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'cold_setup_python_tracing_excluded':tracer is not None and '--cold' in argv,'scope':'diagnostic only; request includes label scoring, NVTX phase/prefill separates first pass'},indent=2))
