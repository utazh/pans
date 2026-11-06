from __future__ import annotations
from types import SimpleNamespace as NS
from pathlib import Path
import copy,json
import numpy as np
import pytest
import torch
from contiguous_fuxian.sharedkv.core import Reader,attention_with_lse
from contiguous_fuxian.sharedkv_v21.ops import pack
from contiguous_fuxian.sharedkv_v25.attention import repaired_attention
from contiguous_fuxian.sharedkv_v26.control import source_proxy,controlled_attention,ControlStats
from contiguous_fuxian.sharedkv_v26.admission import ReuseRegistry,admission_scope
from contiguous_fuxian.sharedkv_v26.analysis import paired_quality,paired_time_blocks,records
from contiguous_fuxian.sharedkv_v26.study import split_blocks,balanced_orders,command,read_bundle

torch.set_num_threads(1)

def tensors(t=3,private=3,dtype=torch.float64):
 torch.manual_seed(12);d=8;h=2
 q=torch.randn(1,t,4,d,dtype=dtype)
 pk,pv=[torch.randn(1,14,h,d,dtype=dtype) for _ in range(2)]
 nk,nv=[torch.randn(1,4,h,d,dtype=dtype) for _ in range(2)]
 tk,tv=[torch.randn(1,private,h,d,dtype=dtype) for _ in range(2)]
 r=Reader(torch.eye(d,dtype=dtype).repeat(h,1,1)+.1*torch.randn(h,d,d,dtype=dtype),
          torch.eye(d,dtype=dtype).repeat(h,1,1)+.1*torch.randn(h,d,d,dtype=dtype),
          torch.randn(h,d,dtype=dtype)*.2,torch.randn(h,d,dtype=dtype)*.2)
 return q,pk,pv,nk,nv,tk,tv,r

@pytest.mark.parametrize('t,private',[(3,3),(1,7)])
@pytest.mark.parametrize('packed',[False,True])
def test_proxy_matches_explicit_affine_shared_attention(t,private,packed):
 q,k,v,nk,nv,tk,tv,r=tensors(t,private)
 stats=ControlStats();before=k.clone()
 got=controlled_attention(repaired_attention,q,k,v,nk,nv,tk,tv,cut=10,reader=r,
      packed=pack(r) if packed else None,repair_values='source-proxy',stats=stats)
 kh=torch.einsum('bnhd,hde->bnhe',k,r.A)+r.bk[None,None]
 vh=torch.einsum('bnhd,hde->bnhe',v,r.B)+r.bv[None,None]
 want=attention_with_lse(q,torch.cat((kh,tk),1),torch.cat((vh,tv),1),causal=True)[0]
 torch.testing.assert_close(got,want,rtol=2e-10,atol=2e-10)
 torch.testing.assert_close(k,before,rtol=0,atol=0)
 assert stats.consumer_calls==1 and stats.max_patch_tokens==4
 assert stats.proxy_elements_materialized==2*1*4*2*8

@pytest.mark.parametrize('packed',[False,True])
def test_native_control_is_bitwise_default(packed):
 q,k,v,nk,nv,tk,tv,r=tensors(dtype=torch.float32)
 kw=dict(cut=10,reader=r,packed=pack(r) if packed else None)
 got=controlled_attention(repaired_attention,q,k,v,nk,nv,tk,tv,**kw)
 want=repaired_attention(q,k,v,nk,nv,tk,tv,**kw)
 torch.testing.assert_close(got,want,rtol=0,atol=0)

@pytest.mark.parametrize('cut,limit',[(0,128),(14,128),(10,3),(-1,128)])
def test_reject_full_or_oversized_proxy(cut,limit):
 _,k,v,_,_,_,_,r=tensors()
 with pytest.raises(ValueError):source_proxy(k,v,cut,r,max_tokens=limit)

def test_proxy_requires_native_buffer():
 q,k,v,nk,nv,tk,tv,r=tensors()
 with pytest.raises(ValueError):controlled_attention(repaired_attention,q,k,v,None,None,tk,tv,
    cut=14,reader=r,repair_values='source-proxy')

def fake_token(key=True,shape=(4,1,2),pos=0):
 return NS(device='disk',file_exist=True,key_exist=key,value_exist=not key,
           key=None,value=None,shape=shape,dtype=np.dtype('float16'),pos_id=pos,
           prefix_cache=NS(device_map=['disk']*16))

@pytest.mark.parametrize('key',[True,False])
def test_reuse_cache_clone_not_request_view(key):
 r=ReuseRegistry();token=fake_token(key=key);full=torch.randn(12,1,2,dtype=torch.float16)
 source=full[4:8];r.register(token,source,key)
 assert r.promote(token,'cpu',allow_cpu_test=True)
 got=token.key if key else token.value
 assert got.untyped_storage().data_ptr()!=full.untyped_storage().data_ptr()
 torch.testing.assert_close(got,source,atol=0,rtol=0)
 full.zero_();assert got.abs().sum()>0
 assert token.device=='cpu' and token.prefix_cache.device_map[0]=='cpu'
 assert r.reused==1 and not r.entries

@pytest.mark.parametrize('bad',['dtype','shape','persisted','packed'])
def test_reuse_reject_incorrect_payload(bad):
 r=ReuseRegistry();t=fake_token();x=torch.zeros(4,1,2,dtype=torch.float16)
 if bad=='dtype':x=x.float()
 if bad=='shape':x=x[:3]
 if bad=='persisted':t.file_exist=False
 if bad=='packed':t.value_exist=True
 with pytest.raises(ValueError):r.register(t,x,True)

def test_non_disk_not_registered():
 r=ReuseRegistry();t=fake_token();t.device='cuda:0'
 r.register(t,torch.zeros(4,1,2,dtype=torch.float16),True)
 assert not r.entries

def test_no_cpu_promotion_in_production():
 r=ReuseRegistry();t=fake_token();r.register(t,torch.zeros(4,1,2,dtype=torch.float16),True)
 assert not r.promote(t,'cpu') and t.device=='disk'
 assert not r.promote(t,'disk')
 r.discard(t);assert not r.entries and r.live_chunk_view_bytes==0

def fake_loader(ids):
 keys=[NS(token=fake_token(shape=(min(4,10-4*i),1,2),pos=i)) for i in range(3)]
 vals=[NS(token=fake_token(False,shape=(min(4,10-4*i),1,2),pos=i+3)) for i in range(3)]
 layer=NS(disk_type='KV_Division',key_tokens=keys,value_tokens=vals)
 loader=NS(_pcache=NS(cache=[NS(layers=[layer])]),_prefix_id=0,chunk_size=4,prefix_tokens=10,
           impress_reordered=False,_config=NS(cache_type='CKLFU'),defer_cache_score_updates=False)
 x=torch.arange(len(ids)*2,dtype=torch.float16).reshape(len(ids),1,2)
 loader._resolve_entry=lambda lid,entry:(x,x+1,torch.tensor(ids))
 return loader,layer,x

def test_register_resolved_full_and_short_chunks():
 loader,layer,x=fake_loader([0,1,2,3,8,9]);r=ReuseRegistry()
 r.register_resolved(loader,0,loader._resolve_entry(0,None))
 assert r.registered==4 and r.entries[id(layer.key_tokens[2].token)].source.shape[0]==2
 assert r.finish()['remaining_entries_released_at_end']==4 and not r.entries

@pytest.mark.parametrize('ids',[[0,1,2],[1,2,3,4],[0,0,1,2,3],[4,5,6,7,9]])
def test_resolved_partial_or_duplicate_rejected(ids):
 loader,layer,x=fake_loader(ids);r=ReuseRegistry()
 with pytest.raises(ValueError):r.register_resolved(loader,0,loader._resolve_entry(0,None))
 r.finish()

def test_duplicate_registration_rejected():
 r=ReuseRegistry();t=fake_token();x=torch.zeros(4,1,2,dtype=torch.float16)
 r.register(t,x,True)
 with pytest.raises(ValueError):r.register(t,x,True)

class DummyToken:
 def to(self,device):self.device=device
class DummyScore:
 def update_score(self,item,new_score):item.token.to('disk')

def test_scope_restores_apis_on_error_and_ownership():
 loader,layer,x=fake_loader([0,1,2,3]);old=loader._resolve_entry
 r=ReuseRegistry();oldto,oldu=DummyToken.to,DummyScore.update_score
 with pytest.raises(RuntimeError):
  with admission_scope(loader,r,vendor=NS(TokenCacheDisk=DummyToken,ScoreUnit=DummyScore)):
   loader._resolve_entry(0,None)
   assert r.registered==2 and loader._resolve_entry is not old
   raise RuntimeError('test')
 assert loader._resolve_entry is old and DummyToken.to is oldto and DummyScore.update_score is oldu

def test_no_instance_bound_cycle_after_scope():
 class Loader:
  def _resolve_entry(self,lid,entry):return None
 l=Loader();l._config=NS(cache_type='CKLFU')
 with admission_scope(l,ReuseRegistry(),vendor=NS(TokenCacheDisk=DummyToken,ScoreUnit=DummyScore)):pass
 assert '_resolve_entry' not in vars(l)

@pytest.mark.parametrize('field,value',[('defer_cache_score_updates',True),('impress_rolling_period_prefetch',True),('impress_period_prefetch_size',2)])
def test_admission_configuration_rejected(field,value):
 l,_,_=fake_loader([0,1,2,3]);setattr(l,field,value)
 with pytest.raises(ValueError):
  with admission_scope(l,ReuseRegistry(),vendor=NS(TokenCacheDisk=DummyToken,ScoreUnit=DummyScore)):pass

def quality(n,h,g):
 # jointly correct enough to construct discordant counts; accuracy values not used to tune a rule
 a={('t',str(i)):{'correct':i<h or i>=h+g} for i in range(n)}
 b={('t',str(i)):{'correct':i>=h} for i in range(n)}
 return a,b

@pytest.mark.parametrize('h,g,expected',[ (96,111,False),(94,91,False),(89,108,False),(109,86,False)])
def test_report_legacy_ni_still_fails(h,g,expected):
 a,b=quality(1536,h,g);r=paired_quality(a,b)
 assert r['legacy_bound_passes']==expected
 assert r['net_delta_pp']==pytest.approx((g-h)/1536*100)

def test_quality_no_flips_not_zero_risk():
 a,b=quality(10,0,0);r=paired_quality(a,b)
 assert r['conservative_lower_pp']<0 and not r['legacy_bound_passes']

def test_quality_pairing_fail():
 a,b=quality(20,1,2);b.pop(next(iter(b)))
 with pytest.raises(ValueError):paired_quality(a,b)

def test_duplicate_record_rejected(tmp_path):
 p=tmp_path/'r.jsonl';p.write_text('{"task":"t","uid":1,"correct":true}\n'*2)
 with pytest.raises(ValueError):records(p)

def test_plan_unique_blocks_and_balanced_orders():
 rows=[{'uid':str(i)} for i in range(400)]
 b=split_blocks(rows,12,32,5)
 assert len({r['uid'] for block in b for r in block})==384
 orders=balanced_orders(12,5)
 for arm in orders[0]:
  assert [sum(order[p]==arm for order in orders) for p in range(4)]==[3,3,3,3]

@pytest.mark.parametrize('n,size',[(2,4),(6,4),(4,30)])
def test_bad_split(n,size):
 with pytest.raises(ValueError):split_blocks([{'uid':i} for i in range(100)],n,size,1)

def test_timing_repeats_not_independent_queries():
 blocks=[]
 for i in range(4):
  a={('t',str(k)):{'ttft_ms':100+i,'correct':True} for k in range(5)}
  b={('t',str(k)):{'ttft_ms':80+i,'correct':True} for k in range(5)}
  blocks.append({'baseline':a,'student':b,'block':i})
 r=paired_time_blocks(blocks,draws=100)
 assert r['unique_query_keys']==5 and r['request_pairs_including_repeats']==20
 assert r['paired_run_blocks']==4 and r['delta_pct']<0

def test_commands_charge_shared_and_keep_cold_and_selector(tmp_path):
 spec={'repo':str(tmp_path),'python':'python','task':'trec','pcache_root':'/cache','adapter_dir':'/adapter','block_size':16}
 argv,out=command(spec,tmp_path/'study',0,'preserve','S-reuse')
 assert argv.count('--')==4
 assert '--diagnostic' not in argv and '--selection-plan' not in argv
 assert argv[argv.index('--admission')+1]=='reuse-prefetch'
 assert argv[argv.index('--mode')+1]=='shared'
 assert argv[argv.index('--read-advice')+1]=='preserve'
 assert '--packed-ops' in argv

def test_bundle_duplicate_and_prefix_checks(tmp_path):
 row={'uid':'a','query_text':'hello','prefix_text':'P','task':'trec'}
 p=tmp_path/'trec.jsonl';p.write_text(json.dumps(row)+'\n'+json.dumps({**row,'uid':'b'})+'\n')
 with pytest.raises(ValueError):read_bundle(p,'trec')
 p.write_text(json.dumps(row)+'\n'+json.dumps({**row,'uid':'b','query_text':'different','prefix_text':'Q'})+'\n')
 with pytest.raises(ValueError):read_bundle(p,'trec')

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable: real cache-promotion path still requires GPU audits')
def test_cuda_promotion_clone_is_exact():
 r=ReuseRegistry();t=fake_token();x=torch.randn(4,1,2,device='cuda',dtype=torch.float16)
 r.register(t,x,True);assert r.promote(t,str(x.device))
 torch.cuda.synchronize();torch.testing.assert_close(t.key,x,rtol=0,atol=0)
 assert t.key.data_ptr()!=x.data_ptr()

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable: no real vLLM parity or latency certification')
def test_cuda_proxy_smoke():
 pytest.importorskip('vllm')
 q,k,v,nk,nv,tk,tv,r=tensors(dtype=torch.float32)
 q,k,v,nk,nv,tk,tv=[x.cuda().bfloat16() for x in (q,k,v,nk,nv,tk,tv)]
 r=r.to(device='cuda',dtype=torch.bfloat16)
 out=controlled_attention(repaired_attention,q,k,v,nk,nv,tk,tv,cut=10,reader=r,
     packed=pack(r),repair_values='source-proxy',backend='vllm2')
 assert bool(torch.isfinite(out).all())

def test_direct_export_preserves_policy_and_does_not_mutate_source(tmp_path):
 from contiguous_fuxian.sharedkv_v26.direct import make_direct,export_direct
 from contiguous_fuxian.sharedkv_v25.policy import FORMAT
 _,_,_,_,_,_,_,r=tensors()
 bundle={'format':FORMAT,'geometry':{'kv_heads':2,'head_dim':8},'sources':[0,0,2,2],
    'native_policy':{'blocks':4,'chunk_size':16,'rule':'last_selected_blocks'},
    'readers':{'1':r.state_dict(),'3':r.state_dict()},'calibration_query_hashes':['x'],
    'diagnostic_only':False}
 before=copy.deepcopy(bundle)
 direct=make_direct(bundle)
 assert direct['native_policy']==before['native_policy'] and direct['sources']==before['sources']
 assert direct['calibration_query_hashes']==['x'] and direct['diagnostic_only']
 torch.testing.assert_close(direct['readers']['1']['A'],torch.eye(8,dtype=torch.float64).repeat(2,1,1))
 torch.testing.assert_close(bundle['readers']['1']['A'],before['readers']['1']['A'])
 src=tmp_path/'original.pt';out=tmp_path/'direct.pt';torch.save(bundle,src);raw=src.read_bytes()
 export_direct(src,out);assert src.read_bytes()==raw
 with pytest.raises(FileExistsError):export_direct(src,out)

def test_quality_report_numerically_matches_v25_preregistered_rule():
 a,b=quality(1536,94,91)
 assert paired_quality(a,b)['conservative_lower_pp']==pytest.approx(-2.982762125175875)

def test_unshared_native_control_preserves_input_storage():
 q,k,v,_,_,tk,tv,_=tensors()
 seen={}
 def capture(*args,**kwargs):seen['args']=args;return args[0]
 controlled_attention(capture,q,k,v,None,None,tk,tv,cut=k.shape[1],reader=None)
 assert seen['args'][1] is k and seen['args'][4] is None

def test_reuse_unknown_token_falls_back_without_modification():
 r=ReuseRegistry();t=fake_token()
 assert not r.promote(t,'cpu',allow_cpu_test=True)
 assert t.device=='disk' and t.key is None and t.value is None

def test_empty_records_rejected(tmp_path):
 p=tmp_path/'x';p.write_text('')
 with pytest.raises(ValueError):records(p)

def test_nonfinite_timing_rejected():
 a={('t','1'):{'ttft_ms':float('nan'),'correct':True}}
 b={('t','1'):{'ttft_ms':2.,'correct':True}}
 with pytest.raises(ValueError):paired_time_blocks([{'baseline':a,'student':b,'block':i} for i in range(4)],draws=20)

def test_integration_patch_restores_install_and_attention_on_error(tmp_path,monkeypatch):
 from contiguous_fuxian.sharedkv_v25 import runtime as online
 from contiguous_fuxian.sharedkv_v26.integration import install_controls
 class Runtime:
  mode='shared'
  def install(self,runner):
   runner.runtime_variant=lambda **kw:'fake'
   runner.greedy_flexgen_completion=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('request failure'))
 oldclass=online.Runtime;monkeypatch.setattr(online,'Runtime',Runtime)
 orig=Runtime.install;attn=online.repaired_attention;runner=NS()
 out=tmp_path/'audit.jsonl'
 with install_controls(audit_path=out):
  Runtime().install(runner)
  with pytest.raises(RuntimeError):runner.greedy_flexgen_completion(loader=NS(_info=NS(task='trec')),query_token_ids=[1,2])
  assert online.repaired_attention is attn
 assert Runtime.install is orig
 row=json.loads(out.read_text());assert row['status']=='failed' and row['repair_values']=='native'

def test_integration_native_passes_through_same_result(tmp_path,monkeypatch):
 from contiguous_fuxian.sharedkv_v25 import runtime as online
 from contiguous_fuxian.sharedkv_v26.integration import install_controls
 result=('ok',1.,2.,{})
 class Runtime:
  mode='identity'
  def install(self,runner):
   runner.runtime_variant=lambda **kw:'fake'
   runner.greedy_flexgen_completion=lambda *a,**kw:result
 monkeypatch.setattr(online,'Runtime',Runtime)
 runner=NS();out=tmp_path/'audit.jsonl'
 with install_controls(audit_path=out):
  Runtime().install(runner)
  assert runner.greedy_flexgen_completion(loader=NS(_info=NS(task='trec')),query_token_ids=[1]) is result
 assert json.loads(out.read_text())['status']=='complete'
