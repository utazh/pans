
from pathlib import Path
import argparse,sys,time,json,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['group','score'],required=True);p.add_argument('--worklog',required=True);p.add_argument('rest',nargs=argparse.REMAINDER);a=p.parse_args()
argv=a.rest[1:] if a.rest and a.rest[0]=='--' else a.rest
import torch
from contiguous_fuxian.sharedkv_v23.block import TrainableReader
initial=[];counts={};old_init=TrainableReader.__init__
def init(self,reader,*args,**kw):
 old_init(self,reader,*args,**kw)
 initial.append({name:hashlib.sha256(getattr(self,name).detach().float().cpu().contiguous().numpy().tobytes()).hexdigest() for name in ['A','B','bk','bv']})
TrainableReader.__init__=init
if a.kind=='group':
 from contiguous_fuxian.sharedkv_v23 import train
 old=train.replay_group
 def counted(*args,**kw):
  key=('grad' if torch.is_grad_enabled() else 'eval')+('_native' if kw.get('native') else '_student')
  counts[key]=counts.get(key,0)+1
  return old(*args,**kw)
 train.replay_group=counted
else:
 from contiguous_fuxian.sharedkv_v24 import train
 old=train.continuation_tokens
 def counted(*args,**kw):
  key='branch_grad' if torch.is_grad_enabled() else 'branch_no_grad'
  counts[key]=counts.get(key,0)+1
  return old(*args,**kw)
 train.continuation_tokens=counted
started=time.perf_counter();status='failed'
try:
 sys.argv=['train',*argv];train.main();status='completed'
finally:
 if torch.cuda.is_available():torch.cuda.synchronize()
 Path(a.worklog).write_text(json.dumps({'kind':a.kind,'status':status,'args':argv,'initial_reader_tensor_sha256':initial,'forward_calls_including_validation':counts,'wall_seconds_total_including_load':time.perf_counter()-started,'cuda_peak_allocated':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,'cuda_peak_reserved':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,'checkpoint_backward_recomputation_not_in_top_level_forward_count':True},indent=2))
 print('WORKLOG',a.worklog,status,flush=True)
