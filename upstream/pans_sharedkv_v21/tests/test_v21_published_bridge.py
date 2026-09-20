import copy,math,types
import pytest
import torch
from contiguous_fuxian.sharedkv.core import Reader
from contiguous_fuxian.sharedkv.runtime import RequestState
from contiguous_fuxian.sharedkv.io import validate_sources
from contiguous_fuxian.sharedkv.selection import fixed_selection
from contiguous_fuxian.sharedkv_v21.backend import LengthPool,attention,_ACTIVE_POOL
from contiguous_fuxian.sharedkv_v21.runtime import schedule_window,shadow_resolve,Runtime,storage_bytes
from contiguous_fuxian.sharedkv_v21.audit import harm_rate_upper,paired_quality
from contiguous_fuxian.sharedkv_v21.bundles import isolate

class Loader:
    def __init__(self):
        self._selected_by_layer=[list(range(5)) for _ in range(4)]
        self._loaded={};self.jobs=[];self.resolves=[];self.scored=0
    def schedule(self,l):self.jobs.append(l)
    def resolve(self,l):
        self.resolves.append(l)
        if l not in self._loaded:self._loaded[l]=(torch.ones(5,2,4)*l,torch.ones(5,2,4)*(l+1))
        return self._loaded[l]
    def configure_impress_layer(self,*,layer,selected_tokens,prefetch_priority_tokens):
        self._selected_by_layer[layer]=selected_tokens


def test_metadata_pool_reuses_identical_lengths_and_bounds_memory():
    p=LengthPool(2)
    a=p.get(1,5,'cpu');b=p.get(1,5,'cpu')
    assert a.data_ptr()==b.data_ptr() and a.tolist()==[0,5]
    p.get(1,8,'cpu');p.get(1,9,'cpu')
    assert len(p.data)==2 and p.hits==1 and p.misses==3
    assert p.stats()['cu_seqlens_resident_bytes']==16

@pytest.mark.parametrize('policy,expected',[('window',[0,1,2]),('demand',[])])
def test_p8_selection_period_decouples_from_prefetch(policy,expected):
    s=RequestState([0,1,2,2],{},'reference');l=Loader()
    schedule_window(s,l,0,4,policy)
    assert l.jobs==expected and l._selected_by_layer==[list(range(5))]*4


def test_shadow_load_restores_consumer_io_but_not_attention_storage():
    s=RequestState([0,1,2,2],{},'reference');l=Loader()
    cfg={'shadow_load':True,'shadow_buffers':{}}
    schedule_window(s,l,0,4,'window',True)
    assert l.jobs==[0,1,2,3]
    pk,_=s.resolve(l,3,dtype=torch.float32,device='cpu')
    shadow_resolve(s,l,3,cfg,torch.float32,'cpu');shadow_resolve(s,l,3,cfg,torch.float32,'cpu')
    assert l.resolves==[2,3] and s.resolve_calls==1
    assert len(cfg['shadow_buffers'])==1 and pk[0,0,0,0]==2
    assert cfg['shadow_buffers'][3][0][0,0,0]==3
    assert s.resolve(l,2,dtype=torch.float32,device='cpu')[0].data_ptr()==pk.data_ptr()


def test_same_ids_enforced_even_in_demand_mode():
    s=RequestState([0,1,2,2],{},'reference');l=Loader();l._selected_by_layer[3]=[0]
    with pytest.raises(ValueError):schedule_window(s,l,0,4,'demand')


def test_fixed_plan_still_runs_scoring_and_restores_method_ownership():
    l=Loader();plan=[list(range(3))]*4
    def real_selector():
        l.scored+=1
        for j in range(4):l.configure_impress_layer(layer=j,selected_tokens=[9],prefetch_priority_tokens=[9])
    with fixed_selection(l,plan):real_selector()
    assert l.scored==1 and l._selected_by_layer==plan
    assert 'configure_impress_layer' not in vars(l)
    real_selector();assert l._selected_by_layer==[[9]]*4

@pytest.mark.parametrize('period',[2,4,8])
def test_last_pair_supported_but_period_requires_separate_calibration(period):
    sources=list(range(28));sources[27]=26
    validate_sources(sources,period=period)


def test_p1_not_compatible_with_protected_leaders():
    sources=list(range(28));sources[27]=26
    with pytest.raises(ValueError):validate_sources(sources,period=1)


def test_constructor_accepts_actual_vllm2_backend_and_candidate_gate():
    r=Runtime(mode='shared',backend='vllm2',adapter_dir='dummy',allow_candidate=True)
    assert r.allow_diagnostic is True
    with pytest.raises(ValueError):Runtime(mode='identity',backend='vllm2',shadow_load=True)


def test_zero_observed_harm_is_not_zero_population_harm():
    assert harm_rate_upper(0,128)==pytest.approx(1-.05**(1/128))
    assert harm_rate_upper(0,128)*100>2
    assert harm_rate_upper(0,300)*100<1
    b=[{'query_hash':str(i),'correct':1} for i in range(128)]
    r=paired_quality(b,b,bootstrap=50)
    assert r['paired_bootstrap_95pct_delta_pp']==[0,0]
    assert not r['conservative_noninferiority_supported']


def test_binomial_bound_matches_scipy_if_present():
    stats=pytest.importorskip('scipy.stats')
    for k,n in [(1,128),(13,128),(10,20)]:
        assert harm_rate_upper(k,n)==pytest.approx(stats.beta.ppf(.95,k+1,n-k),rel=1e-10)


def test_storage_accounting_counts_views_once_and_extra_buffers():
    a=torch.ones(3,4);b=a.T
    assert storage_bytes([a,b])==48
    assert storage_bytes([a,b,b.contiguous()])==96


def test_isolated_bundle_retains_bans_and_marks_diagnostic():
    eye=torch.eye(4).repeat(2,1,1)
    r=Reader(eye+.1,eye+.2,torch.ones(2,4),torch.ones(2,4)).state_dict()
    b={'sources':[0,1,2,2],'readers':{'3':r},'period':4,'calibration_query_hashes':['a','b'],'report':{'passed':True}}
    x=isolate(b,[3],direct=True)
    assert x['sources']==b['sources'] and x['diagnostic_only']
    assert x['calibration_query_hashes']==['a','b'] and x['accepted_consumers']==0
    assert torch.equal(x['readers']['3']['A'],eye)
    assert torch.equal(b['readers']['3']['A'],eye+.1) # no mutation

@pytest.mark.skipif(not torch.cuda.is_available(),reason='No CUDA available')
def test_gpu_published_vllm_backend_cache_parity():
    pytest.importorskip('vllm.vllm_flash_attn')
    from contiguous_fuxian.sharedkv.core import attention_with_lse
    q=torch.randn(1,3,28,128,device='cuda',dtype=torch.bfloat16)*.2
    k=torch.randn(1,19,4,128,device='cuda',dtype=torch.bfloat16)*.2;v=torch.randn_like(k)
    p=LengthPool();tok=_ACTIVE_POOL.set(p)
    try:
        a,z=attention(q,k,v,causal=True,backend='vllm2')
        b,w=attention_with_lse(q,k,v,causal=True,backend='vllm2')
        torch.testing.assert_close(a,b,atol=.003,rtol=.01);torch.testing.assert_close(z,w,atol=.003,rtol=.01)
        attention(q,k,v,causal=True,backend='vllm2');assert p.hits==2
    finally:_ACTIVE_POOL.reset(tok)
