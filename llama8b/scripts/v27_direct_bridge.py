
from v26_common import *
import torch
from contiguous_fuxian.sharedkv_v24.records import load,FIELDS
from contiguous_fuxian.sharedkv.io import read_prefix
from contiguous_fuxian.sharedkv_v27.baseline_quality import FORMAT
from contiguous_fuxian.sharedkv_v27.common import torch_new
fit=load(ROOT/'runtime_data/v24_teacher_fit','trec','fit');head=next(iter(fit.values()))
bundle=dict(format=FORMAT,**{k:head[k] for k in FIELDS},calibration_query_hashes=sorted(fit),method='direct-map-prefix-quality',quality_only=True,latency_valid=False,hardware_bytes_valid=False,position_offset=0,rope_theta=1e6,sources=list(range(28)),native={str(i):[read_prefix(ROOT/'runtime_data/prefix_store','trec',i,k)[0] for k in ['key','value']] for i in range(28)})
torch_new(ROOT/'assets/v27_direct_identity/trec.pt',bundle)
name='v27_direct_identity_bridge'
a=argv(name,gpu=3,n=16,diagnostic=True,admission='reuse-prefetch',split='full_eval')
a[a.index('--pcache-root')+1]=str(ROOT/'runtime_data/v27_pcache_gpu3')
end=a.index('--task')-1;a[end:end]=['--logits',str(ROOT/f'runtime_data/{name}_logits'),'--selection-plan',str(ROOT/'runtime_data/v27_fixed2_I_logits')]
run(name,[PY,'-u','scripts/run_baseline_quality_v27.py','--representation',str(ROOT/'assets/v27_direct_identity/trec.pt'),'--',*a[3:]],3)
