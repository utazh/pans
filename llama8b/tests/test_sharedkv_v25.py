from __future__ import annotations
import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from contiguous_fuxian.sharedkv_v25.policy import NativePolicy,partition,payload_account,FORMAT
from contiguous_fuxian.sharedkv_v25.attention import repaired_attention
from contiguous_fuxian.sharedkv_v25.geometry import inspect_config
from contiguous_fuxian.sharedkv.core import Reader,attention_with_lse
from contiguous_fuxian.sharedkv_v21.ops import pack

torch.set_num_threads(1)

@pytest.mark.parametrize('r,cut',[(0,14),(1,12),(2,8)])
def test_partition_short_last_block(r,cut):
 ids=list(range(8))+list(range(12,18))
 p=partition(ids,18,NativePolicy(r,4))
 assert p.cut==cut and p.shared_ids+p.native_ids==tuple(ids)
 assert not set(p.shared_ids)&set(p.native_ids)

@pytest.mark.parametrize('ids',[[0,1,1,2],[1,2,3],[0,1,2,4],[0,1,2,3,20]])
def test_partition_rejects_invalid_or_partial(ids):
 with pytest.raises(ValueError):partition(ids,20,NativePolicy(1,4))

@pytest.mark.parametrize('r',[-1,1.5,True])
def test_bad_policy(r):
 with pytest.raises(ValueError):NativePolicy(r)

def test_partition_forbids_silent_full_fallback():
 with pytest.raises(ValueError):partition(list(range(8)),8,NativePolicy(2,4))

def test_accounting_includes_native_patches():
 ids=list(range(16));a=payload_account([ids]*4,[0,0,2,2],16,2,4,NativePolicy(1,4))
 assert a['full_selected_source_count']==2 and a['partial_native_layer_count']==2
 assert a['total_selected_payload_bytes']==(32+8)*32
 assert a['selected_payload_reduction_fraction']==.375

def test_accounting_rejects_mismatched_ids():
 with pytest.raises(ValueError):payload_account([list(range(16)),list(range(12))],[0,0],16,2,4,NativePolicy(1,4))


def rand_reader(h,d,dtype=torch.float64):
 return Reader(torch.eye(d,dtype=dtype).repeat(h,1,1)+torch.randn(h,d,d,dtype=dtype)*.1,
               torch.eye(d,dtype=dtype).repeat(h,1,1)+torch.randn(h,d,d,dtype=dtype)*.1,
               torch.randn(h,d,dtype=dtype)*.2,torch.randn(h,d,dtype=dtype)*.2)

@pytest.mark.parametrize('t,private',[(3,3),(1,6)])
@pytest.mark.parametrize('packed',[False,True])
def test_exact_algebra_and_joint_normalization(t,private,packed):
 torch.manual_seed(1);D=8;H=2
 q=torch.randn(1,t,4,D,dtype=torch.float64)
 k,v=[torch.randn(1,12,H,D,dtype=torch.float64) for _ in range(2)]
 nk,nv=[torch.randn(1,4,H,D,dtype=torch.float64) for _ in range(2)]
 tk,tv=[torch.randn(1,private,H,D,dtype=torch.float64) for _ in range(2)]
 r=rand_reader(H,D)
 got=repaired_attention(q,k,v,nk,nv,tk,tv,cut=8,reader=r,packed=pack(r) if packed else None)
 kh=torch.einsum('bnhd,hde->bnhe',k[:,:8],r.A)+r.bk[None,None]
 vh=torch.einsum('bnhd,hde->bnhe',v[:,:8],r.B)+r.bv[None,None]
 want=attention_with_lse(q,torch.cat((kh,nk,tk),1),torch.cat((vh,nv,tv),1),causal=True)[0]
 torch.testing.assert_close(got,want,atol=2e-10,rtol=2e-10)

def test_all_native_mathematical_endpoint():
 torch.manual_seed(5);q=torch.randn(1,2,4,8,dtype=torch.float64)
 k,v,nk,nv=[torch.randn(1,4,2,8,dtype=torch.float64) for _ in range(4)]
 tk,tv=[torch.randn(1,2,2,8,dtype=torch.float64) for _ in range(2)]
 got=repaired_attention(q,k,v,nk,nv,tk,tv,cut=0,reader=rand_reader(2,8))
 want=attention_with_lse(q,torch.cat((nk,tk),1),torch.cat((nv,tv),1),causal=True)[0]
 torch.testing.assert_close(got,want)

def test_no_prefix_copy_in_consumer(monkeypatch):
 import contiguous_fuxian.sharedkv_v21.ops as ops
 q=torch.randn(1,2,4,8);k=torch.randn(1,12,2,8);v=k.clone();n=k[:,:4].clone()
 seen={}
 def observe(q,pk,pv,tk,tv,**kw):seen.update(k=pk,v=pv);return q
 monkeypatch.setattr(ops,'dispatch',observe)
 repaired_attention(q,k,v,n,n,q[:,:,:2],q[:,:,:2],cut=8,reader=rand_reader(2,8,torch.float32))
 assert seen['k'].untyped_storage().data_ptr()==k.untyped_storage().data_ptr()
 assert seen['v'].untyped_storage().data_ptr()==v.untyped_storage().data_ptr()
 assert seen['k'].shape[1]==8

class FakeLoader:
 def __init__(self):
  self.prefix_tokens=16;self._selected_by_layer=[list(range(16)) for _ in range(4)]
  self.calls=[];self.score=[];self._cache_update_ms=0.;self._cache_score_updates=0
  self._cache_score_updated_layers=set();self.defer_cache_score_updates=False
  self._cache_score_updater=lambda l,ids,_:self.score.append((l,ids)) or len(ids)//4
 def schedule(self,lid):self.calls.append(('full',lid,list(self._selected_by_layer[lid])))
 def _submit_positions(self,lid,ids,pending,**kwargs):
  self.calls.append(('patch',lid,ids));k=torch.full((len(ids),1,2),float(lid))
  pending[lid]=(SimpleNamespace(result=lambda:(k,k.clone(),torch.tensor(ids))),0.)
 def _resolve_entry(self,lid,entry):return entry[0].result()
class FakeState:
 sources=[0,2,2,2]
 def schedule_window(self,loader,start,end):
  for s in sorted(set(self.sources[start:end])):loader.schedule(s)

def test_actual_repair_requests_and_scores_do_not_restore_full_consumer():
 from contiguous_fuxian.sharedkv_v25.runtime import RepairManager
 l=FakeLoader();original=copy.deepcopy(l._selected_by_layer)
 m=RepairManager(FakeState(),l,NativePolicy(1,4));m.configure(0,4,'window')
 assert [x[:2] for x in l.calls]==[('full',0),('full',2),('patch',1),('patch',3)]
 for lid in (1,3):
  k,v,cut=m.resolve(lid,torch.float32,'cpu');assert cut==12 and k.shape[1]==4
  m.resolve(lid,torch.float32,'cpu')
 assert m.resolve_calls==2 and l._selected_by_layer==original
 assert l.score==[(1,[12,13,14,15]),(3,[12,13,14,15])]
 assert not m.pending

def test_repair_demand_and_empty_policy():
 from contiguous_fuxian.sharedkv_v25.runtime import RepairManager
 l=FakeLoader();m=RepairManager(FakeState(),l,NativePolicy(0,4));m.configure(0,4,'demand')
 assert not l.calls
 assert m.resolve(1,torch.float32,'cpu')==(None,None,16)

def test_pending_drained_on_error():
 from contiguous_fuxian.sharedkv_v25.runtime import RepairManager
 l=FakeLoader();m=RepairManager(FakeState(),l,NativePolicy(1,4));m.configure(0,4,'window');m.drain()
 assert not m.pending

@pytest.mark.parametrize('L,Q,H,D',[(28,28,4,128),(48,40,8,128)])
def test_qwen_geometry(L,Q,H,D):
 g=inspect_config(dict(model_type='qwen2',num_hidden_layers=L,num_attention_heads=Q,num_key_value_heads=H,hidden_size=Q*D))
 assert g['probe_query_heads']==list(range(0,Q,Q//H))
 assert g['fp16_prefix_bytes_per_token']==4*L*H*D

@pytest.mark.parametrize('bad',[dict(model_type='llama'),dict(model_type='qwen2',rope_scaling={'x':1}),dict(model_type='qwen2',use_sliding_window=True)])
def test_bad_model_geometry(bad):
 with pytest.raises(ValueError):inspect_config(bad)

class Rotary(nn.Module):
 def forward(self,h,pos):
  f=torch.tensor([.1,.01],device=h.device);a=pos[...,None]*f
  a=torch.cat((a,a),-1)
  return a.cos().to(h.dtype),a.sin().to(h.dtype)
class Layer(nn.Module):
 def __init__(self):
  super().__init__();self.input_layernorm=nn.LayerNorm(8);self.post_attention_layernorm=nn.LayerNorm(8)
  self.self_attn=nn.Module();self.self_attn.config=SimpleNamespace(num_attention_heads=2,num_key_value_heads=1,hidden_size=8)
  self.self_attn.head_dim=4
  for name,size in [('q_proj',8),('k_proj',4),('v_proj',4),('o_proj',8)]:
   setattr(self.self_attn,name,nn.Linear(8,size,bias=name!='o_proj'))
  self.mlp=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,8))
class ToyModel(nn.Module):
 def __init__(self):
  super().__init__();self.model=nn.Module();self.model.embed_tokens=nn.Embedding(32,8)
  self.model.layers=nn.ModuleList([Layer() for _ in range(4)]);self.model.rotary_emb=Rotary()
  self.model.norm=nn.LayerNorm(8);self.lm_head=nn.Linear(8,32,bias=False)
  self.requires_grad_(False);self.eval()

def fixture():
 from contiguous_fuxian.sharedkv_v23.block import TrainableReader
 torch.manual_seed(91);model=ToyModel()
 row={'query_token_ids':[2,3,5],'label_ids':{'a':[6,7,8],'b':[9]},'prefix_tokens':12,
      'selection_by_layer':[list(range(8)) for _ in range(4)],'period':8,'geometry':{'layers':4},
      'teacher_backend':'vllm2','teacher_layout':'hybrid'}
 sources=[0,0,2,2];raw={i:tuple(torch.randn(12,1,4) for _ in range(2)) for i in range(4)}
 readers=nn.ModuleDict({str(l):TrainableReader(rand_reader(1,4,torch.float32),ste=False) for l in (1,3)})
 return model,row,sources,raw,readers

def test_r0_matches_published_faithful_replay():
 from contiguous_fuxian.sharedkv_v25.replay import selected_bank,continuation
 from contiguous_fuxian.sharedkv_v24.faithful import continuation as old
 m,row,s,raw,r=fixture();bank=selected_bank(raw,row,s,'cpu',torch.float32,NativePolicy(0,2))
 for label in row['label_ids']:
  torch.testing.assert_close(continuation(m,row,label,s,bank,r,checkpoint_layers=False),
                             old(m,row,label,s,bank,r,checkpoint_layers=False),atol=0,rtol=0)

@pytest.mark.parametrize('ckpt',[False,True])
def test_multiconsumer_gradients_and_private_decode_lengths(ckpt):
 from contiguous_fuxian.sharedkv_v25.replay import selected_bank,decoder,continuation
 m,row,s,raw,r=fixture();bank=selected_bank(raw,row,s,'cpu',torch.float32,NativePolicy(1,2))
 lp=continuation(m,row,'a',s,bank,r,checkpoint_layers=ckpt);lp.sum().backward()
 assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in reader.parameters()) for reader in r.values())
 assert all(p.grad is None for p in m.parameters())
 with torch.no_grad():
  _,cache=decoder(m,torch.tensor([[2,3,5]]),12,None,s,bank,r,False)
  assert all(k.shape[1]==3 for k,v in cache)
  _,cache2=decoder(m,torch.tensor([[6]]),15,cache,s,bank,r,False)
  assert all(k.shape[1]==4 for k,v in cache2)
  _,cache3=decoder(m,torch.tensor([[9]]),15,cache,s,bank,r,False)
  assert all(k.shape[1]==4 for k,v in cache3)
  assert all(k.shape[1]==3 for k,v in cache)
  assert bank.repairs[1][0].shape[1]==2

def test_streamed_backward_equal_to_full_graph():
 from contiguous_fuxian.sharedkv_v25.replay import selected_bank,continuation
 from contiguous_fuxian.sharedkv_v24.replay import streamed_backward,objective
 m,row,s,raw,r=fixture();r2=copy.deepcopy(r);bank=selected_bank(raw,row,s,'cpu',torch.float32,NativePolicy(1,2))
 labels=['a','b'];teacher=[torch.tensor([-2.,-1.,-3.]),torch.tensor([-2.])]
 got=[continuation(m,row,x,s,bank,r,checkpoint_layers=False) for x in labels]
 objective(got,teacher)[0].backward()
 streamed_backward(lambda x:continuation(m,row,x,s,bank,r2,checkpoint_layers=False),labels,teacher,{})
 for a,b in zip(r.parameters(),r2.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5)

def test_training_patch_restores_globals_and_writes_distinct_format(tmp_path):
 from contiguous_fuxian.sharedkv_v24 import train as t
 from contiguous_fuxian.sharedkv_v25.train import training_patch
 old=t.selected_bank
 with training_patch(NativePolicy(1,2)):
  assert t.selected_bank is not old
  p=tmp_path/'x.pt';t.save_new(p,{'readers':{},'format':'old','diagnostic_only':False})
  saved=torch.load(p,weights_only=True)
  assert saved['format']==FORMAT and saved['diagnostic_only'] and saved['native_policy']['blocks']==1
 assert t.selected_bank is old

def test_v24_teacher_bank_ignores_repairs_when_native():
 from contiguous_fuxian.sharedkv_v25.replay import selected_bank
 m,row,s,raw,r=fixture();b=selected_bank(raw,row,list(range(4)),'cpu',torch.float32,NativePolicy(1,2))
 assert not b.repairs and len(b)==4

def test_numerical_teacher_gate_unchanged_by_wrapper():
 from contiguous_fuxian.sharedkv_v25.train import training_patch
 from contiguous_fuxian.sharedkv_v24 import train as t
 original=t.evaluate
 with training_patch(NativePolicy(1,2)):assert t.evaluate is original

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable; actual vLLM attention must be checked on target server')
def test_cuda_partitioned_attention():
 pytest.importorskip('vllm')
 q=torch.randn(1,3,28,128,device='cuda',dtype=torch.bfloat16)
 k,v=[torch.randn(1,80,4,128,device='cuda',dtype=q.dtype) for _ in range(2)]
 nk,nv=[torch.randn(1,16,4,128,device='cuda',dtype=q.dtype) for _ in range(2)]
 tk,tv=[torch.randn(1,3,4,128,device='cuda',dtype=q.dtype) for _ in range(2)]
 r=rand_reader(4,128,torch.float32).to(device='cuda',dtype=q.dtype)
 got=repaired_attention(q,k,v,nk,nv,tk,tv,cut=64,reader=r,packed=pack(r),backend='vllm2')
 want=repaired_attention(q,k,v,nk,nv,tk,tv,cut=64,reader=r,packed=pack(r),backend='reference')
 torch.testing.assert_close(got,want,atol=.06,rtol=.06)

def test_full_decoder_bridge_matches_training_replay(monkeypatch):
 import sys,types
 from contiguous_fuxian.sharedkv_v25 import runtime as online,replay
 from contiguous_fuxian.sharedkv import runtime as old
 from contiguous_fuxian.sharedkv_v21 import runtime as old21
 m,row,s,raw,readers=fixture();policy=NativePolicy(1,2)
 class Loader(FakeLoader):
  def __init__(self):
   super().__init__();self.prefix_tokens=12;self._selected_by_layer=[list(range(8)) for _ in range(4)]
   self._loaded={};self.full_calls=[];self.chunk_size=2
  def resolve(self,lid):
   if lid not in self._loaded:
    self.full_calls.append(lid);self._loaded[lid]=tuple(x[:8] for x in raw[lid])
   return self._loaded[lid]
  def prefetch_selector_keys(self,lid):pass
  def _submit_positions(self,lid,ids,pending,**kw):
   self.calls.append(('patch',lid,ids));vals=tuple(x[ids] for x in raw[lid])
   pending[lid]=(SimpleNamespace(result=lambda:(*vals,torch.tensor(ids))),0)
 loader=Loader()
 mock=types.ModuleType('contiguous_fuxian.flexgen_qwen_reprefill')
 mock.configure_online_layer_selection=lambda **kw:kw['impress_selection_period_size']
 monkeypatch.setitem(sys.modules,'contiguous_fuxian.flexgen_qwen_reprefill',mock)
 class Cache:
  def __init__(self):self.data={}
  def get_seq_length(self,l):return self.data[l][0].shape[2] if l in self.data else 0
  def update(self,k,v,lid,kw):
   if lid in self.data:
    a,b=self.data[lid];k,v=torch.cat((a,k),2),torch.cat((b,v),2)
   self.data[lid]=(k,v);return k,v
 cache=Cache();rr={int(l):Reader(*(getattr(r,n).detach() for n in ('A','B','bk','bv'))) for l,r in readers.items()}
 state=old.RequestState(s,rr,'reference',geometry=dict(layers=4,q_heads=2,kv_heads=1,head_dim=4))
 cfg=dict(sink=None,replay=None,shadow_load=False,nvtx=False,selector_calls=0,schedule='window',
          packed_ops=True,packed={lid:pack(r) for lid,r in rr.items()},layout='hybrid')
 a=old._ACTIVE.set(state);b=old21._SETTINGS.set(cfg);c=online._POLICY.set(policy);extra={};d=online._EXTRA.set(extra)
 try:
  with torch.no_grad():
   actual=online.decoder_logits(model=m,input_ids=torch.tensor([row['query_token_ids']]),cache=cache,
           position_start=12,loader=loader,period_size=8,subperiod_size=4,impress_selection_period_size=8)
   bank=replay.selected_bank(raw,row,s,'cpu',torch.float32,policy)
   expected,private=replay.decoder(m,torch.tensor([row['query_token_ids']]),12,None,s,bank,readers,False)
   torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
   actual2=online.decoder_logits(model=m,input_ids=torch.tensor([[6]]),cache=cache,position_start=15,
              loader=loader,period_size=8,subperiod_size=4,impress_selection_period_size=8)
   expected2,_=replay.decoder(m,torch.tensor([[6]]),15,private,s,bank,readers,False)
   torch.testing.assert_close(actual2,expected2,atol=2e-6,rtol=2e-6)
  assert loader.full_calls==[0,2] and extra['manager'].resolve_calls==2
  assert cfg['selector_calls']==1 and all(cache.get_seq_length(l)==4 for l in range(4))
 finally:
  online._EXTRA.reset(d);online._POLICY.reset(c);old21._SETTINGS.reset(b);old._ACTIVE.reset(a)

def test_training_evaluate_one_step_export(tmp_path):
 from contiguous_fuxian.sharedkv_v25.replay import selected_bank,continuation
 from contiguous_fuxian.sharedkv_v24.replay import streamed_backward
 from contiguous_fuxian.sharedkv_v25.train import training_patch
 from contiguous_fuxian.sharedkv_v24 import train as trainer
 m,row,s,raw,readers=fixture();policy=NativePolicy(1,2);native=list(range(4))
 nb=selected_bank(raw,row,native,'cpu',torch.float32,policy)
 row['label_order']=['a','b']
 row['teacher_token_logprobs']={l:continuation(m,row,l,native,nb,{},checkpoint_layers=False).detach().tolist() for l in ['a','b']}
 args=SimpleNamespace(max_sequence=32,temperature=1.,score_weight=1.,token_weight=.1,rank_weight=.25,margin_cap=.5)
 with training_patch(policy):
  parity=trainer.evaluate(m,{'q':row},raw,s,readers,args,native_parity=True)
  assert parity['max_abs_token_logprob_error']==0
  bank=trainer.selected_bank(raw,row,s,'cpu',torch.float32)
  opt=torch.optim.Adam(readers.parameters(),lr=1e-4);opt.zero_grad()
  loss,_,_=streamed_backward(lambda l:trainer.continuation_tokens(m,row,l,s,bank,readers,checkpoint_layers=True),
                        ['a','b'],trainer.teacher_vectors(row,'cpu'),trainer.settings(args))
  assert torch.isfinite(torch.tensor(loss));opt.step()
  metric=trainer.evaluate(m,{'q':row},raw,s,readers,args)
  assert metric['requests']==1 and not metric['gold_accuracy_evaluated']
  path=tmp_path/'candidate.pt';trainer.save_new(path,{'readers':{l:r.export().state_dict() for l,r in readers.items()},'report':metric})
  assert torch.load(path,weights_only=True)['format']==FORMAT
