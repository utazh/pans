
from v26_common import *
for shared in [False,True]:
 for mode in ['original','reuse-prefetch']:
  name='v26_probe_'+('S' if shared else 'I')+'_'+mode
  run(name,argv(name,shared=shared,admission=mode,n=2,diagnostic=True),3)
