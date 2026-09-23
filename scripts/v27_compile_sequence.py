
from v26_common import *
import torch
from contiguous_fuxian.sharedkv_v27.baseline_math import rotary,factorize,reconstruct
from contiguous_fuxian.sharedkv.io import read_prefix
raw=[read_prefix(ROOT/'runtime_data/prefix_store','trec',i,'key')[0][:96].cuda() for i in range(4)]
pos=torch.arange(96,device='cuda')
pre=[rotary(x,pos,theta=1e6,inverse=True) for x in raw]
f=factorize(pre,96,exact=True);errors=[]
for i,x in enumerate(raw):
 y=rotary(reconstruct(f,i,pos),pos,theta=1e6)
 errors.append(float(torch.linalg.vector_norm(y-x)/torch.linalg.vector_norm(x)))
assert max(errors)<1e-5
(ROOT/'audit/v27_real_prefix_roundtrip.json').write_text(json.dumps({'relative_l2':errors,'tokens':96,'layers':4,'full_rank':96}))
del raw,pre,f;torch.cuda.empty_cache()
for tag,rk,rv,exact in [('full',2048,2048,True),('r256_384',256,384,False),('r384_576',384,576,False)]:
 cmd=[PY,'-u','scripts/compile_prefix_baseline_v27.py','--method','clf','--teacher-fit',str(ROOT/'runtime_data/v24_teacher_fit'),'--store-root',str(ROOT/'runtime_data/prefix_store'),'--task','trec','--model-config',MODEL+'/config.json','--window','4','--rank-k',str(rk),'--rank-v',str(rv),'--device','cuda','--output',str(ROOT/f'assets/v27_clf_{tag}/trec.pt')]
 if exact:cmd+=['--exact-svd']
 run('v27_compile_'+tag,cmd,3)
(ROOT/'audit/v27_compile_done.json').write_text(json.dumps({'passed':True}))
