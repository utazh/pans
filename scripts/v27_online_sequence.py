
from v26_common import *
assert (ROOT/'audit/v27_fixed_validation.json').exists()
for arm in ['I','native','proxy','r0']:
 name='v27_online_'+arm
 cmd=argv(name,gpu=2,n=500,shared=arm!='I',proxy=arm=='proxy',diagnostic=True,admission='reuse-prefetch',adapter=ROOT/'assets/v27_frozen_r0' if arm=='r0' else ROOT/'assets/v25_g16_r4_s24',split='full_eval')
 cmd[cmd.index('--pcache-root')+1]=str(ROOT/'runtime_data/v27_pcache_gpu2')
 run(name,cmd,2)
(ROOT/'audit/v27_online_done.json').write_text(json.dumps({'n':500,'arms':['I','native','proxy','r0']}))
