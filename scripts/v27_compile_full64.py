
from v26_common import *
import torch,copy
from contiguous_fuxian.sharedkv.io import read_prefix
from contiguous_fuxian.sharedkv_v27.baseline_math import rotary,factorize,factor_bytes
from contiguous_fuxian.sharedkv_v27.common import torch_new
torch.set_num_threads(4)
old=torch.load(ROOT/'assets/v27_clf_full/trec.pt',weights_only=True)
meta={k:v for k,v in old.items() if k!='groups'};del old
groups=[];pos=torch.arange(meta['prefix_tokens']);start=time.perf_counter()
for lo in range(0,28,4):
 layers=list(range(lo,lo+4))
 ks=[rotary(read_prefix(ROOT/'runtime_data/prefix_store','trec',i,'key')[0].double(),pos,theta=meta['rope_theta'],inverse=True) for i in layers]
 vs=[read_prefix(ROOT/'runtime_data/prefix_store','trec',i,'value')[0].double() for i in layers]
 fk=factorize(ks,2048,exact=True);fv=factorize(vs,2048,exact=True)
 groups.append(dict(layers=layers,key=fk,value=fv));print('full64',lo,flush=True)
meta.update(groups=groups,compile_seconds=time.perf_counter()-start,diagnostic_precision='float64 full rank reference',factor_bytes=sum(factor_bytes(g[k]) for g in groups for k in ['key','value']))
torch_new(ROOT/'assets/v27_clf_full64/trec.pt',meta)
(ROOT/'audit/v27_full64_ready.json').write_text(json.dumps({'compile_seconds':meta['compile_seconds']}))
