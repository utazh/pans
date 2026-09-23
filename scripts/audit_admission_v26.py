
"""Actual-GPU diagnostic only; asserts cache values/ownership, heaps and private KV."""
from pathlib import Path
import argparse,json,runpy,sys,hashlib,torch
from contextvars import ContextVar
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from contiguous_fuxian.sharedkv_v25.runtime import Runtime
from contiguous_fuxian.sharedkv_v26.admission import ReuseRegistry
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('rest',nargs=argparse.REMAINDER);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
argv=a.rest[1:] if a.rest[:1]==['--'] else a.rest
scope=ContextVar('v26_audit',default=None)
oldinstall=Runtime.install;oldpromote=ReuseRegistry.promote;oldregister=ReuseRegistry.register;oldfinish=ReuseRegistry.finish
def backing(registry):
 seen={}
 for en in registry.entries.values():
  st=en.source.untyped_storage();seen[(en.source.device.type,en.source.device.index,st.data_ptr())]=st.nbytes()
 return sum(seen.values())
def register(self,*aa,**kw):
 z=oldregister(self,*aa,**kw);st=scope.get()
 if st is not None:st['peak_registry_backing_bytes']=max(st['peak_registry_backing_bytes'],backing(self))
 return z
def promote(self,token,device,**kw):
 en=self.entries.get(id(token));z=oldpromote(self,token,device,**kw);st=scope.get()
 if z:
  actual=token.key if en.key else token.value
  assert torch.equal(actual,en.source)
  assert actual.untyped_storage().data_ptr()!=en.source.untyped_storage().data_ptr()
  assert actual.untyped_storage().nbytes()==actual.numel()*actual.element_size()
  assert str(token.prefix_cache.device_map[token.pos_id])==str(device)
  if st is not None:st['promotions_verified']+=1
 return z
def finish(self):
 z=oldfinish(self);assert not self.entries and self.live_chunk_view_bytes==0
 return z
def snapshot(loader):
 pc=loader._pcache;ctl=pc.control;loc={};counts={'cuda:0':0.,'cpu':0.};data=hashlib.sha256();descriptors=[]
 for pid,prefix in enumerate(pc.cache):
  for lid,layer in enumerate(prefix.layers):
   for kind,items in [('k',layer.key_tokens),('v',layer.value_tokens),('head',[layer.head_token])]:
    for idx,item in enumerate(items):
     token=item.token;identity=(pid,lid,kind,idx)
     loc[id(item)]=identity;loc[id(item.control)]=identity
     if token.pos_id is not None:assert str(layer.device_map[token.pos_id])==str(token.device)
     descriptors.append([*identity,str(token.device),list(item.score)])
     if str(token.device) in counts:
      counts[str(token.device)]+=item.size
      tensor=token.key if kind in ('k','head') else token.value
      data.update(json.dumps(identity).encode());data.update(tensor.detach().cpu().contiguous().numpy().tobytes())
     else:assert token.key is None and token.value is None
 heaps={}
 for dev,heap in [('cuda:0',ctl.gpu_heap),('cpu',ctl.cpu_heap)]:
  rows=[]
  for i,item in enumerate(heap._queue):
   assert item.heap is heap and item.heap_pos==i
   assert str(item.token.device)==dev
   if i:assert heap._queue[(i-1)//2].score[heap.score_pos]<=item.score[heap.score_pos]
   rows.append(loc[id(item)])
  assert len(set(rows))==len(rows);heaps[dev]=rows
  expected=ctl.gpu_num if dev=='cuda:0' else ctl.cpu_num
  assert abs(counts[dev]-expected)<1e-6,(dev,counts[dev],expected)
  assert expected<=(ctl.gpu_size if dev=='cuda:0' else ctl.cpu_size)+1e-6
 return dict(cache_state_sha256=hashlib.sha256(json.dumps(descriptors).encode()).hexdigest(),resident_values_sha256=data.hexdigest(),heap_order=heaps,gpu_mb=ctl.gpu_num,cpu_mb=ctl.cpu_num)
def install(self,runner):
 oldinstall(self,runner);previous=runner.greedy_flexgen_completion;decoder=runner.flexgen_sparse_decoder_logits
 def decode(*aa,**kw):
  result=decoder(*aa,**kw);st=scope.get()
  if st is not None:
   layers=kw['cache'].layers;keys=[x.keys.untyped_storage().data_ptr() for x in layers];values=[x.values.untyped_storage().data_ptr() for x in layers]
   assert len(layers)==28 and len(set(keys+values))==56
   lens=[x.keys.shape[2] for x in layers];assert len(set(lens))==1
   if 'first_logits' not in st:st['first_logits']=result[:,-1].detach().cpu()
   st['decoder_calls']+=1
  return result
 def completion(*aa,**kw):
  st=dict(promotions_verified=0,peak_registry_backing_bytes=0,decoder_calls=0);token=scope.set(st)
  scores=kw.get('label_token_logprobs_out')
  if scores is None:scores={};kw['label_token_logprobs_out']=scores
  torch.cuda.reset_peak_memory_stats()
  try:
   result=previous(*aa,**kw);loader=kw['loader'];st.update(snapshot(loader))
   st.update(selection_by_layer=loader._selected_by_layer,label_token_logprobs=scores,query_token_ids=kw['query_token_ids'],peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),all28_private=True)
   index=len(list(out.glob('request*.pt')))
   with (out/f'request{index:03d}.pt').open('xb') as f:torch.save(st,f)
   return result
  finally:scope.reset(token)
 runner.flexgen_sparse_decoder_logits=decode;runner.greedy_flexgen_completion=completion
Runtime.install=install;ReuseRegistry.promote=promote;ReuseRegistry.register=register;ReuseRegistry.finish=finish
try:
 sys.argv=[str(ROOT/'scripts/run_evidence_v26.py'),*argv];runpy.run_path(sys.argv[0],run_name='__main__')
finally:Runtime.install=oldinstall;ReuseRegistry.promote=oldpromote;ReuseRegistry.register=oldregister;ReuseRegistry.finish=oldfinish
