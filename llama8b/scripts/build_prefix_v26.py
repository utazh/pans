
from v26_common import *
import gc,torch
from contiguous_fuxian.sparse_qwen_reprefill import prepare_prefix_store
from contiguous_fuxian.flexgen_pcache import FlexGenPcacheConfig,FlexGenPcacheStore
from contiguous_fuxian.quantized_key_index import build_quantized_key_index
start=time.time();torch.cuda.reset_peak_memory_stats()
store=ROOT/'runtime_data/v26_prefix_p1';assert not store.exists()
info=prepare_prefix_store(model_path=MODEL,bundle_dir=ROOT/'data/v26/p1/eval',store_root=store,tasks=['sst2','subj','trec','rte'],samples_per_task=1,device='cuda',dtype='bfloat16')
gc.collect();torch.cuda.empty_cache();prefill=time.time()-start
cache=ROOT/'runtime_data/v26_p1_pcache_gpu3';assert not cache.exists()
pc=FlexGenPcacheStore(FlexGenPcacheConfig(flexgen_root=ROOT/'vendor/flexgen',kv_dir=cache,chunk_size=16,gpu_cache_mb=0,cpu_cache_mb=0,cache_type='LRU',selector_kv_head_ids=(0,1,2,3)))
for task in ['sst2','subj','trec','rte']:pc.add_task(store_root=store,task=task)
for p in cache.iterdir():
 if p.is_file():
  with p.open('rb') as f:os.fsync(f.fileno())
report=dict(prefix_prefill_wall=prefill,store_and_pcache_wall=time.time()-start,peak_gpu=torch.cuda.max_memory_allocated(),prefix_info={k:vars(v) for k,v in info.items()})
with (ROOT/'audit/v26_prefix_build.json').open('x') as f:json.dump(report,f,indent=2)
print('PREFIX BUILD DONE',report,flush=True)
