
from v26_common import *
import random
# Remain inside this foreground-authorized experiment; no recurring automation.
for marker in ['v27_baseline_quality_done.json','v27_bf16_done.json']:
 while not (ROOT/'audit'/marker).exists():time.sleep(10)
assert (ROOT/'audit/v27_online_done.json').exists()
orders=[['I','native','r0'],['native','r0','I'],['r0','I','native']]
for block,order in enumerate(orders):
 for arm in order:
  mem=subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
  if int(mem.strip())>500:raise RuntimeError('GPU2 became occupied; no interference with others')
  name=f'v27_cold_b{block}_{arm}'
  a=argv(name,gpu=2,n=32,warm=4,cold='preserve',shared=arm!='I',admission='reuse-prefetch',adapter=ROOT/'assets/v27_frozen_r0' if arm=='r0' else ROOT/'assets/v25_g16_r4_s24')
  a[a.index('--pcache-root')+1]=str(ROOT/'runtime_data/v27_pcache_gpu2')
  a[a.index('--owned-cache-root')+1]=str(ROOT/'runtime_data/v27_pcache_gpu2')
  a[a.index('--bundle-dir')+1]=str(ROOT/f'data/v27/cold_b{block}')
  run(name,a,2)
(ROOT/'audit/v27_cold_done.json').write_text(json.dumps({'blocks':3,'queries_per_block':32,'arms':['I','native','r0'],'formal_cold_requests':288,'scope':'three disjoint request blocks, Latin-square order; exploratory, not broad speed certification'}))
