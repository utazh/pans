
"""Diagnostic actual source-view, native-value, cache-admission, private-KV audit."""
from pathlib import Path
from contextvars import ContextVar
import argparse,json,runpy,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
p=argparse.ArgumentParser();p.add_argument('--audit-output',required=True);p.add_argument('rest',nargs=argparse.REMAINDER);a=p.parse_args()
argv=a.rest[1:] if a.rest[:1]==['--'] else a.rest
out=Path(a.audit_output);assert not out.exists()
from contiguous_fuxian.sharedkv_v25 import runtime as rt
from contiguous_fuxian.sharedkv_v25.policy import partition
from contiguous_fuxian.sharedkv_v21 import ops
from contiguous_fuxian.sharedkv import runtime as v1
from contiguous_fuxian.sharedkv.io import query_hash,read_prefix
scope=ContextVar('v25_contract_scope',default=None);dispatch_scope=ContextVar('v25_dispatch_scope',default=None)
old_install=rt.Runtime.install;old_attention=rt.repaired_attention;old_dispatch=ops.dispatch;raw={}
def dispatch(q,pk,pv,tk,tv,*aa,**kw):
 expected=dispatch_scope.get()
 if expected is not None:
  assert (pk.data_ptr(),pv.data_ptr())==expected['source_ptrs']
  assert pk.shape[1]==expected['cut'] and tk.shape[1]==expected['tail_length']
  scope.get()['verified_backend_source_view_calls']+=1
 return old_dispatch(q,pk,pv,tk,tv,*aa,**kw)
def attention(q,pk,pv,nk,nv,tk,tv,**kw):
 st=scope.get()
 if st is None:return old_attention(q,pk,pv,nk,nv,tk,tv,**kw)
 lid=st['layer_calls']%len(st['sources']);st['layer_calls']+=1
 active=v1._ACTIVE.get();src=st['sources'][lid]
 assert (pk.data_ptr(),pv.data_ptr())==(active.bank[src][0].data_ptr(),active.bank[src][1].data_ptr())
 pcy=st['policy'];ids=st['loader']._selected_by_layer[lid]
 part=partition(ids,st['loader'].prefix_tokens,pcy) if src!=lid else None
 expected_n=len(part.native_ids) if part else 0
 assert (0 if nk is None else nk.shape[1])==expected_n
 if part:assert list(part.shared_ids)+list(part.native_ids)==list(ids) and not set(part.shared_ids)&set(part.native_ids)
 if expected_n and lid not in st['native_values_checked']:
  if lid not in raw:raw[lid]=tuple(read_prefix(ROOT/'runtime_data/prefix_store','trec',lid,k)[0] for k in ['key','value'])
  index=torch.tensor(part.native_ids,dtype=torch.long)
  expected=[x.index_select(0,index).to(device=q.device,dtype=q.dtype).unsqueeze(0) for x in raw[lid]]
  assert torch.equal(nk,expected[0]) and torch.equal(nv,expected[1]),('native value mismatch',lid)
  assert nk.data_ptr()!=pk.data_ptr() and nv.data_ptr()!=pv.data_ptr()
  st['native_values_checked'].add(lid)
 cut=part.cut if part else pk.shape[1]
 token=dispatch_scope.set({'source_ptrs':(pk.data_ptr(),pv.data_ptr()),'cut':cut,'tail_length':tk.shape[1]+expected_n})
 try:return old_attention(q,pk,pv,nk,nv,tk,tv,**kw)
 finally:dispatch_scope.reset(token)
def install(runtime,runner):
 old_install(runtime,runner);previous=runner.greedy_flexgen_completion;decode=runner.flexgen_sparse_decoder_logits
 def decoder(*aa,**kw):
  result=decode(*aa,**kw);st=scope.get()
  if st is None:return result
  layers=kw['cache'].layers;kp=[x.keys.untyped_storage().data_ptr() for x in layers];vp=[x.values.untyped_storage().data_ptr() for x in layers];lengths=[x.keys.shape[2] for x in layers]
  assert len(layers)==28 and len(set(kp))==28 and len(set(vp))==28 and not set(kp)&set(vp)
  assert len(set(lengths))==1 and st['query_tokens']<=lengths[0]<=st['query_tokens']+st['max_tokens']+4
  st['decoder_calls']+=1;st['max_private_growth']=max(st['max_private_growth'],lengths[0]-st['query_tokens'])
  return result
 def completion(*aa,**kw):
  loader=kw['loader'];prepared=runtime._prepare(kw['model'],loader,kw['impress_selection_period_size']);sources=prepared[1];policy=runtime.policies[loader._info.task]
  st={'query_hash':query_hash(kw['query_token_ids']),'query_tokens':len(kw['query_token_ids']),'max_tokens':kw['max_tokens'],'sources':sources,'policy':policy,'loader':loader,'layer_calls':0,'decoder_calls':0,'max_private_growth':0,'verified_backend_source_view_calls':0,'native_values_checked':set(),'admissions':[]}
  updater=loader._cache_score_updater
  assert callable(updater), 'Expected active online cache-admission callback'
  def update(lid,ids,*aa,**kk):
   logical=loader._selected_by_layer[lid]
   expected=logical if sources[lid]==lid else list(partition(logical,loader.prefix_tokens,policy).native_ids)
   assert list(ids)==list(expected),('unexpected cache admission',lid,len(ids),len(expected))
   st['admissions'].append({'layer':lid,'tokens':len(ids),'consumer':sources[lid]!=lid})
   return updater(lid,ids,*aa,**kk)
  token=scope.set(st);loader._cache_score_updater=update
  try:
   result=previous(*aa,**kw)
   assert st['layer_calls']==st['decoder_calls']*28 and st['verified_backend_source_view_calls']==st['layer_calls']
   expected={i for i,s in enumerate(sources) if i!=s and policy.blocks>0}
   assert st['native_values_checked']==expected
   record={k:v for k,v in st.items() if k not in ['loader','policy','native_values_checked']}
   record.update(native_policy=policy.state(),native_values_checked=sorted(expected),all_28_private=True,support_partition_exact=True,native_values_equal_stored_layer=True,all_backend_prefixes_source_views=True,cache_admission_only_native_repairs=True,formal_timing_eligible=False)
   with out.open('a') as f:f.write(json.dumps(record)+'\n')
   return result
  finally:loader._cache_score_updater=updater;scope.reset(token)
 runner.flexgen_sparse_decoder_logits=decoder;runner.greedy_flexgen_completion=completion
rt.Runtime.install=install;rt.repaired_attention=attention;ops.dispatch=dispatch
try:
 sys.argv=[str(ROOT/'scripts/run_native_prefix_v25.py'),*argv];runpy.run_path(sys.argv[0],run_name='__main__')
finally:rt.Runtime.install=old_install;rt.repaired_attention=old_attention;ops.dispatch=old_dispatch
