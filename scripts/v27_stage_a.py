
from v26_common import *
from contiguous_fuxian.sharedkv_v27.frozen_control import export_frozen_r0
import torch
parent=ROOT/'assets/v25_g16_r4_s24/trec.pt'
out=ROOT/'assets/v27_frozen_r0/trec.pt'
export_frozen_r0(parent,out)
a=torch.load(parent,weights_only=True);b=torch.load(out,weights_only=True)
assert a['sources']==b['sources']
assert all(torch.equal(v,b['readers'][k][n]) for k,r in a['readers'].items() for n,v in r.items() if torch.is_tensor(v))
for arm in ['I','native','proxy','r0']:
 name='v27_fixed_'+arm
 cmd=argv(name,gpu=2,n=16,shared=arm!='I',proxy=arm=='proxy',diagnostic=True,admission='reuse-prefetch',adapter=out.parent if arm=='r0' else parent.parent,split='full_eval')
 end=cmd.index('--task')-1
 cmd[end:end]=['--logits',str(ROOT/f'runtime_data/{name}_logits')]
 if arm!='I':
  end=cmd.index('--task')-1;cmd[end:end]=['--selection-plan',str(ROOT/'runtime_data/v27_fixed_I_logits')]
 run(name,cmd,2)
(ROOT/'audit/v27_fixed_done.json').write_text(json.dumps({'arms':['I','native','proxy','r0'],'n':16}))
