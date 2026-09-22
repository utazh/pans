"""Actual GPU private-cache pointer audit, diagnostic decode only."""
from pathlib import Path
from contextvars import ContextVar
import argparse,json,runpy,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
p=argparse.ArgumentParser();p.add_argument('--audit-output',required=True);p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
argv=a.args[1:] if a.args and a.args[0]=='--' else a.args
out=Path(a.audit_output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
from contiguous_fuxian.sharedkv_v25.runtime import Runtime
from contiguous_fuxian.sharedkv.io import query_hash
active=ContextVar('v25_private_audit',default=None);old_install=Runtime.install
def install(self,runner):
 old_install(self,runner)
 old_decode=runner.flexgen_sparse_decoder_logits;old_greedy=runner.greedy_flexgen_completion
 def decoder(*args,**kwargs):
  result=old_decode(*args,**kwargs);state=active.get()
  if state is not None:
   cache=kwargs['cache'];layers=cache.layers
   kp=[v.keys.untyped_storage().data_ptr() for v in layers]
   vp=[v.values.untyped_storage().data_ptr() for v in layers]
   lengths=[v.keys.shape[2] for v in layers]
   assert len(layers)==28 and len(set(kp))==28 and len(set(vp))==28
   assert not set(kp)&set(vp)
   assert len(set(lengths))==1 and state['query_tokens']<=lengths[0]<=state['query_tokens']+state['max_tokens']+4
   row={**state,'position_start':int(kwargs['position_start']),'private_lengths':lengths,'key_storage_ptrs':kp,'value_storage_ptrs':vp,'all_layers_private':True}
   with out.open('a') as f:f.write(json.dumps(row)+'\n')
  return result
 def completion(*args,**kwargs):
  token=active.set({'query_hash':query_hash(kwargs['query_token_ids']),'query_tokens':len(kwargs['query_token_ids']),'max_tokens':kwargs['max_tokens']})
  try:return old_greedy(*args,**kwargs)
  finally:active.reset(token)
 runner.flexgen_sparse_decoder_logits=decoder;runner.greedy_flexgen_completion=completion
Runtime.install=install
try:
 sys.argv=[str(ROOT/'scripts/run_native_prefix_v25.py'),*argv]
 runpy.run_path(sys.argv[0],run_name='__main__')
finally:Runtime.install=old_install
