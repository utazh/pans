
from v26_common import *
for cold in [None,'preserve']:
 for shared in [False,True]:
  for mode in ['original','reuse-prefetch']:
   name='v26_audit3_'+('cold' if cold else 'warm')+'_'+('S' if shared else 'I')+'_'+mode
   base=argv(name,shared=shared,admission=mode,n=8,warm=2,cold=cold,diagnostic=True)
   cmd=[PY,'-u','scripts/audit_admission_v26.py','--output',str(ROOT/f'audit/{name}'),'--',*base[3:]]
   run(name,cmd,3)
