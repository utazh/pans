from pathlib import Path
from types import SimpleNamespace
import copy,json
import pytest
import torch
from contiguous_fuxian.sharedkv_v27.algebra import frozen_r0,proxy_split,affine_kv,attn
from contiguous_fuxian.sharedkv_v27.frozen_control import make_frozen_r0,export_frozen_r0
from contiguous_fuxian.sharedkv_v27.common import V25_FORMAT,validate_sources,torch_new
from contiguous_fuxian.sharedkv_v27.topology import bounded_plans,describe
from contiguous_fuxian.sharedkv_v27.baseline_math import factorize,reconstruct,rotary,factor_bytes,kvsharer_style_search,dissimilar_edges
from contiguous_fuxian.sharedkv_v27.quality import paired_quality,load_records,pareto
from contiguous_fuxian.sharedkv_v27.baseline_quality import QualityMemory,FORMAT

def tensors(t=3,n=7,extra=2,hq=4,hk=2,d=4,device='cpu'):
 gen=torch.Generator(device=device).manual_seed(271)
 rand=lambda *s:torch.randn(*s,generator=gen,dtype=torch.float64,device=device)
 q=rand(1,t,hq,d);k,v=rand(1,n,hk,d),rand(1,n,hk,d)
 tk,tv=rand(1,t+extra,hk,d),rand(1,t+extra,hk,d)
 A,B=rand(hk,d,d),rand(hk,d,d);bk,bv=rand(hk,d),rand(hk,d)
 return q,k,v,tk,tv,A,B,bk,bv

@pytest.mark.parametrize('t,extra,hq,hk',[(1,0,2,2),(1,5,4,2),(3,0,4,2),(3,3,4,1),(5,2,6,2)])
@pytest.mark.parametrize('cut',[1,4,6])
def test_proxy_is_same_affine_memory_in_real_arithmetic(t,extra,hq,hk,cut):
 x=tensors(t=t,extra=extra,hq=hq,hk=hk)
 a=frozen_r0(*x);b=proxy_split(*x,cut=cut)
 torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
 q,k,v,tk,tv,A,B,bk,bv=x
 kk,vv=affine_kv(k,v,A,B,bk,bv)
 c,_=attn(q,torch.cat([kk,tk],1),torch.cat([vv,tv],1),True)
 torch.testing.assert_close(a,c,rtol=1e-11,atol=1e-11)

def test_key_bias_cannot_be_dropped():
 x=list(tensors());a=frozen_r0(*x);x[-2]=torch.zeros_like(x[-2]);b=frozen_r0(*x)
 assert (a-b).abs().max()>1e-3

@pytest.mark.parametrize('cut',[0,7,-1,8])
def test_proxy_bad_cut(cut):
 with pytest.raises(ValueError):proxy_split(*tensors(),cut=cut)

def bundle():
 g={'layers':4,'q_heads':4,'kv_heads':2,'head_dim':4}
 r={'A':torch.randn(2,4,4),'B':torch.randn(2,4,4),'bk':torch.randn(2,4),'bv':torch.randn(2,4)}
 return {'format':V25_FORMAT,'geometry':g,'period':4,'sources':[0,2,2,2],
 'readers':{'1':copy.deepcopy(r),'3':copy.deepcopy(r)},'native_policy':{'blocks':1,'chunk_size':4,'rule':'last_selected_blocks'},
 'calibration_query_hashes':['fit','val'],'fit_query_hashes':['fit'],'validation_query_hashes':['val'],
 'report':{'accuracy':.9},'audited_runtime_base':'unchanged'}

def test_export_only_changes_policy_no_tensor_values():
 b=bundle();o=make_frozen_r0(b)
 assert b['native_policy']['blocks']==1 and o['native_policy']['blocks']==0
 assert o['sources']==b['sources'] and o['calibration_query_hashes']==b['calibration_query_hashes']
 assert 'report' not in o and o['parent_quality_report']==b['report']
 for lid in b['readers']:
  for name in b['readers'][lid]:torch.testing.assert_close(b['readers'][lid][name],o['readers'][lid][name],rtol=0,atol=0)
 assert o['accepted_consumers']==0 and not o['quality_certified']

@pytest.mark.parametrize('problem',['format','r0','heads','reader'])
def test_export_fail_closed(problem):
 b=bundle()
 if problem=='format':b['format']='no'
 if problem=='r0':b['native_policy']['blocks']=0
 if problem=='heads':b['head_maps']={'1':[0,1]}
 if problem=='reader':b['readers']['1']['A'][0,0,0]=float('nan')
 with pytest.raises(ValueError):make_frozen_r0(b)

def test_export_exclusive(tmp_path):
 src=tmp_path/'src.pt';out=tmp_path/'out.pt';torch_new(src,bundle());export_frozen_r0(src,out)
 assert out.exists()
 with pytest.raises(FileExistsError):export_frozen_r0(src,out)

def rows(n=8):return [{'split':'fit','selection_by_layer':[[1,2,3] for _ in range(n)]}]

def test_bounded_topology_even_window():
 result=bounded_plans(rows(),8,8,max_distance=1,max_consumers=2,protect_first=0)
 assert 4 in result
 for g,r in result.items():
  validate_sources(r['sources'],8)
  assert len(set(r['sources']))==g
  assert max(abs(i-s) for i,s in enumerate(r['sources']))<=1
  from collections import Counter
  assert max(Counter(s for i,s in enumerate(r['sources']) if i!=s).values(),default=0)<=2

def test_infeasible_budget_not_widened():
 r=rows();r[0]['selection_by_layer']=[[i] for i in range(8)]
 result=bounded_plans(r,8,8)
 assert list(result)==[8]

def test_all_fit_rows_checked():
 a=rows();b=rows();b[0]['selection_by_layer']=[[i] for i in range(8)]
 assert list(bounded_plans(a+b,8,8))==[8]

def test_protect_and_leaders():
 result=bounded_plans(rows(28),28,8,protect_first=4)
 assert 16 in result
 for r in result.values():
  s=r['sources'];assert all(s[i]==i for i in [0,1,2,3,8,16,24])

def test_only_fit_design():
 r=rows();r[0]['split']='validation'
 with pytest.raises(ValueError):bounded_plans(r,8,8)

@pytest.mark.parametrize('kw',[{'period':9},{'max_distance':0},{'max_consumers':0},{'protect_first':9}])
def test_bad_plan_params(kw):
 args=dict(n=8,period=8);args.update(kw)
 with pytest.raises(ValueError):bounded_plans(rows(),**args)

@pytest.mark.parametrize('theta',[10000.,1000000.])
def test_rope_inverse(theta):
 x=torch.randn(7,2,4,dtype=torch.float64);p=torch.arange(37,44)
 y=rotary(rotary(x,p,theta),p,theta,True)
 torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)

def test_full_rank_clf_selected_reconstruction():
 torch.manual_seed(2);xs=[torch.randn(12,2,3,dtype=torch.float64) for _ in range(2)]
 f=factorize(xs,12,exact=True);idx=torch.tensor([0,3,7,11])
 for i,x in enumerate(xs):torch.testing.assert_close(reconstruct(f,i,idx),x[idx],rtol=1e-11,atol=1e-11)
 assert factor_bytes(f)==sum(t.numel()*t.element_size() for t in [f['left'],*f['rights']])

def test_low_rank_reproducible_and_rng_preserved():
 torch.manual_seed(3);xs=[torch.randn(20,2,4) for _ in range(2)];state=torch.get_rng_state().clone()
 f=factorize(xs,3,seed=7);assert torch.equal(torch.get_rng_state(),state)
 g=factorize(xs,3,seed=7)
 torch.testing.assert_close(f['left'],g['left'],rtol=0,atol=0)
 assert f['rank']==3

@pytest.mark.parametrize('rank',[0,30])
def test_clf_invalid_rank(rank):
 with pytest.raises(ValueError):factorize([torch.randn(5,2,2)],rank)

def test_kvsharer_search_not_fixed_unit_mapping():
 e=[(9.,0,3),(8.,0,2),(7.,1,2)]
 seen=[]
 def evaluate(s):seen.append(s.copy());return .2 if s[3]==0 else .9
 r=kvsharer_style_search(e,4,3,evaluate)
 assert r['target_reached'] and r['sources']==[0,1,0,3] and len(seen)==2

def test_kvsharer_infeasible_is_reported():
 r=kvsharer_style_search([(9.,0,2)],3,1,lambda s:.5,threshold=.5)
 assert not r['target_reached'] and r['achieved_sources']==3

def test_dissimilarity_order():
 raw={i:(torch.full((4,1,2),float(i)),torch.full((4,1,2),float(i))) for i in range(3)}
 e=dissimilar_edges(raw);assert e[0][1:]==(0,2)

def test_quality_bounds_not_degenerate_on_no_flips():
 b={str(i):{'correct':i%2==0,'task':'t'} for i in range(16)}
 r=paired_quality(b,b)
 assert r['delta_pp']==0 and r['conservative_lower_pp']<0 and r['certified']==False

def test_quality_identity_mismatch():
 with pytest.raises(ValueError):paired_quality({'a':{'correct':1,'task':'a'}},{'a':{'correct':1,'task':'b'}})

def test_quality_counts_and_sign():
 b={str(i):{'correct':i<3} for i in range(6)};s={str(i):{'correct':i in (0,2,3,4)} for i in range(6)}
 r=paired_quality(b,s);assert (r['harmful'],r['beneficial'])==(1,2);assert r['delta_pp']==pytest.approx(100/6)

def test_duplicate_quality_uid_rejected(tmp_path):
 p=tmp_path/'a.jsonl';p.write_text('{"uid":"1","correct":true}\n'*2)
 with pytest.raises(ValueError):load_records(p)

def pts():
 common=dict(task='t',prefix_id='p0',model_id='m',eval_uid_digest='u',cache_condition='cold',admission='reuse')
 return [{**common,'name':'I','accuracy':.9,'ttft_ms':100,'subsystem_bytes':100},
         {**common,'name':'S','accuracy':.89,'ttft_ms':60,'subsystem_bytes':70},
         {**common,'name':'bad','accuracy':.8,'ttft_ms':80,'subsystem_bytes':90}]

def test_pareto_not_cherry_pick():assert [c['name'] for c in pareto(pts())]==['I','S']
def test_pareto_mixed_protocol():
 p=pts();p[1]['admission']='original'
 with pytest.raises(ValueError):pareto(p)

def test_quality_memory_direct_is_not_claimed_io():
 k=torch.randn(8,2,4);v=torch.randn_like(k)
 b={'format':FORMAT,'method':'direct-map-prefix-quality','task':'t','token_hash':'h','prefix_tokens':8,'keep_ratio':.25,
 'model_identity':{'x':1},'geometry':{'layers':3},'sources':[0,0,2],'native':{'0':[k,v],'2':[k+1,v+1]}}
 loader=SimpleNamespace(_info=SimpleNamespace(task='t',token_hash='h'),prefix_tokens=8,keep_ratio=.25,layers=3,
 _selector_index=SimpleNamespace(payload={'model_identity':{'x':1}}))
 memory=QualityMemory(b);got=memory.get(loader,1,[0,5],torch.device('cpu'),torch.float32)
 torch.testing.assert_close(got[0][0],k[[0,5]])
 loader.keep_ratio=.5
 with pytest.raises(ValueError):memory.get(loader,1,[0,5],torch.device('cpu'),torch.float32)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='No CUDA; actual BF16/runtime equivalence still requires server')
def test_cuda_proxy_roundoff_only_diagnostic():
 x=tuple(t.to(torch.bfloat16) for t in tensors(device='cuda'))
 a=frozen_r0(*x);b=proxy_split(*x,cut=4)
 assert torch.isfinite(a).all() and torch.isfinite(b).all()
 # Explicitly NOT bitwise equivalence or a real-model quality guarantee.

def synthetic_records_and_store(tmp_path):
 from contiguous_fuxian.sharedkv_v24.records import FORMAT as RF,digest_ids,json_digest,save_new
 root=tmp_path/'store';taskroot=root/'t';taskroot.mkdir(parents=True)
 n,h,d,layers=8,2,4,4;g={'layers':layers,'q_heads':4,'kv_heads':h,'head_dim':d}
 meta={'dtype':'bfloat16','layout':'token,kv_head,head_dim','prefix_tokens':n,'kv_heads':h,'head_dim':d,'layers':layers,'token_hash':'test_prefix'}
 (taskroot/'metadata.json').write_text(json.dumps(meta))
 original={}
 for lid in range(layers):
  kv=[]
  for kind in ('key','value'):
   x=torch.randn(n,h,d).to(torch.bfloat16);x.view(torch.uint16).numpy().tofile(taskroot/f'layer_{lid:02d}_{kind}.bf16')
   kv.append(x.to(torch.float16).to(torch.bfloat16).float())
  original[lid]=kv
 ids=[11,12];qh=digest_ids(ids);plan=[list(range(4)) for _ in range(layers)]
 row={'format':RF,'task':'t','token_hash':'test_prefix','prefix_tokens':n,'geometry':g,'period':4,'keep_ratio':.5,
 'model_identity':{'test':'model'},'teacher_sources':list(range(layers)),'query_hash':qh,'query_token_ids':ids,
 'label_order':['a','b'],'label_ids':{'a':[1],'b':[2,3]},'teacher_token_logprobs':{'a':[-1.],'b':[-2.,-3.]},
 'teacher_scores':{'a':-1.,'b':-2.5},'score_rule':'mean_continuation_logprob','selection_by_layer':plan,
 'selection_hash':json_digest(plan),'split':'fit'}
 fit=tmp_path/'fit';save_new(fit/'t'/f'{qh}.pt',row)
 cfg={'model_type':'qwen2','num_hidden_layers':layers,'num_attention_heads':4,'num_key_value_heads':h,'hidden_size':16,'rope_theta':1e6}
 cpath=tmp_path/'config.json';cpath.write_text(json.dumps(cfg))
 return root,fit,cpath,row,original


def test_synthetic_store_to_clf_to_quality_memory(tmp_path):
 from contiguous_fuxian.sharedkv_v27.compile_baseline import main
 root,fit,cfg,row,raw=synthetic_records_and_store(tmp_path)
 out=tmp_path/'clf.pt'
 main(['--method','clf','--teacher-fit',str(fit),'--store-root',str(root),'--task','t','--model-config',str(cfg),
       '--output',str(out),'--rank-k','8','--rank-v','8','--window','2','--exact-svd'])
 data=torch.load(out,weights_only=True);assert data['quality_only'] and not data['latency_valid']
 loader=SimpleNamespace(_info=SimpleNamespace(task='t',token_hash=row['token_hash']),prefix_tokens=8,keep_ratio=.5,layers=4,
 _selector_index=SimpleNamespace(payload={'model_identity':row['model_identity']}))
 m=QualityMemory(data)
 for lid in range(4):
  result=m.get(loader,lid,[1,4,7],torch.device('cpu'),torch.float32)
  for x,y in zip(result,raw[lid]):torch.testing.assert_close(x[0],y[[1,4,7]],rtol=1e-5,atol=1e-5)


def test_direct_map_compiler_requires_fit_provenance(tmp_path):
 from contiguous_fuxian.sharedkv_v27.compile_baseline import main
 root,fit,cfg,row,raw=synthetic_records_and_store(tmp_path)
 p=tmp_path/'map.json';p.write_text(json.dumps({'sources':[0,0,2,2]}))
 argv=['--method','direct-map','--teacher-fit',str(fit),'--store-root',str(root),'--task','t','--model-config',str(cfg),
       '--output',str(tmp_path/'direct.pt'),'--direct-plan',str(p)]
 with pytest.raises(ValueError,match='provenance'):main(argv)
 p.write_text(json.dumps({'sources':[0,0,2,2],'design_query_hashes':[row['query_hash']]}));main(argv)
 data=torch.load(tmp_path/'direct.pt',weights_only=True)
 assert sorted(data['native'])==['0','2'] and data['sources']==[0,0,2,2]


def test_quality_bridge_preserves_private_scope_and_restores_patch(tmp_path,monkeypatch):
 from contiguous_fuxian.sharedkv_v27.baseline_quality import install_quality_memory
 from contiguous_fuxian.sharedkv.runtime import RequestState
 k=torch.randn(8,2,4);v=torch.randn_like(k)
 b={'format':FORMAT,'method':'direct-map-prefix-quality','task':'t','token_hash':'h','prefix_tokens':8,'keep_ratio':.5,
 'model_identity':{'x':1},'geometry':{'layers':3},'sources':[0,0,2],'native':{'0':[k,v],'2':[k+1,v+1]}}
 calls=[]
 def original(state,loader,layer,*,dtype,device):
  calls.append(layer);state.bank[layer]=(torch.zeros(1,4,2,4),torch.zeros(1,4,2,4));return state.bank[layer]
 monkeypatch.setattr(RequestState,'resolve',original)
 loader=SimpleNamespace(_info=SimpleNamespace(task='t',token_hash='h'),prefix_tokens=8,keep_ratio=.5,layers=3,
 _selected_by_layer=[[0,2,4,6]]*3,_loaded={},_selector_index=SimpleNamespace(payload={'model_identity':{'x':1}}))
 state=RequestState(list(range(3)),{},'reference');cache_sentinel=object();state.private_test=cache_sentinel
 with pytest.raises(RuntimeError,match='simulated'):
  with install_quality_memory(b):
   a=state.resolve(loader,1,dtype=torch.float32,device=torch.device('cpu'))
   z=state.resolve(loader,1,dtype=torch.float32,device=torch.device('cpu'))
   assert a[0].data_ptr()==z[0].data_ptr() and calls==[1]
   torch.testing.assert_close(a[0][0],k[[0,2,4,6]])
   assert state.private_test is cache_sentinel
   raise RuntimeError('simulated request error')
 assert RequestState.resolve is original
