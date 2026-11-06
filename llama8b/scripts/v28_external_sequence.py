from pathlib import Path
import os,json,time,copy,torch,subprocess
from contiguous_fuxian.sharedkv_v28.kvsharer import compile_cell
from contiguous_fuxian.sharedkv_v28.execute import run_external
from contiguous_fuxian.sharedkv_v28.common import sha,write_json,assert_hashes
r=Path('/home/panzihang/src/prism_layerkv');os.chdir(r);torch.set_num_threads(4)
m=json.loads((r/'studies/v28_frozen/manifest.json').read_text());assert_hashes(m['input_sha256'])
m['spec']['gpu']='3';m['spec']['pcache_root']=str(r/'runtime_data/v27_pcache_gpu3')
write_json(r/'audit/v28_external_resource_overlay.json',dict(base_manifest_sha256=sha(r/'studies/v28_frozen/manifest.json'),gpu='3',pcache_root=m['spec']['pcache_root'],scope='Disjoint external preparation/quality jobs; no timing allowed. Inputs and numerical protocol unchanged.'))
cells=sorted(m['cells'],key=lambda c:(c['id']!='trec_b25',c['task'],c['budget']))
for c in cells:
 start=time.time();print('COMPILE_KVSHARER',c['id'],flush=True)
 compile_cell(c,m['spec'],m['root'])
 write_json(r/f"audit/v28_kvsharer_{c['id']}.json",dict(seconds=time.time()-start))
 # Factors are prepared independently on CPU, never during timing.
 receipt=r/f"audit/v28_factors_{c['id']}.json"
 while not receipt.exists():
  if not Path('/proc/'+str(json.loads((r/'audit/v28_factors_process.json').read_text())['pid'])).exists():
   raise RuntimeError('Factor process ended before this cell completed')
  time.sleep(5)
 print('EXTERNAL_QUALITY',c['id'],flush=True)
 run_external(m,c,True)
 write_json(r/f"audit/v28_external_{c['id']}_done.json",dict(completed=time.time()))
write_json(r/'audit/v28_external_done.json',dict(completed=time.time()))
