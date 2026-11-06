
from pathlib import Path
import argparse,sys,time,json,hashlib
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'src'))
p=argparse.ArgumentParser();p.add_argument('--native-blocks',type=int,required=True);p.add_argument('--worklog',required=True);p.add_argument('rest',nargs=argparse.REMAINDER);a=p.parse_args()
argv=a.rest[1:] if a.rest[:1]==['--'] else a.rest
import torch
from contiguous_fuxian.sharedkv_v25.guard import check
from contiguous_fuxian.sharedkv_v25.train import training_patch
from contiguous_fuxian.sharedkv_v25.policy import NativePolicy
from contiguous_fuxian.sharedkv_v23.block import TrainableReader
check(r);initial=[];counts={};old_init=TrainableReader.__init__
def init(self,reader,*aa,**kw):
 old_init(self,reader,*aa,**kw)
 initial.append({n:hashlib.sha256(getattr(self,n).detach().float().cpu().contiguous().numpy().tobytes()).hexdigest() for n in ['A','B','bk','bv']})
TrainableReader.__init__=init
status='failed';started=time.perf_counter()
try:
 with training_patch(NativePolicy(a.native_blocks)) as trainer:
  old=trainer.continuation_tokens
  def counted(*aa,**kw):
   k='branch_grad' if torch.is_grad_enabled() else 'branch_no_grad';counts[k]=counts.get(k,0)+1
   return old(*aa,**kw)
  trainer.continuation_tokens=counted
  sys.argv=['train',*argv];trainer.main();status='completed'
finally:
 if torch.cuda.is_available():torch.cuda.synchronize()
 d={'status':status,'native_blocks':a.native_blocks,'args':argv,'initial_reader_tensor_sha256':initial,'forward_calls_including_validation':counts,'wall_seconds_total_including_load':time.perf_counter()-started,'cuda_peak_allocated':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,'cuda_peak_reserved':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,'checkpoint_backward_recomputation_not_in_top_level_forward_count':True}
 with Path(a.worklog).open('x') as f:json.dump(d,f,indent=2)
 print('WORKLOG',status,a.worklog,flush=True)
