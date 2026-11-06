from __future__ import annotations
import copy,importlib.util,json,sys,types
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from contiguous_fuxian.sharedkv.core import Reader
from contiguous_fuxian.sharedkv_v23.block import TrainableReader
from contiguous_fuxian.sharedkv_v24.records import *
from contiguous_fuxian.sharedkv_v24.replay import *
from contiguous_fuxian.sharedkv_v24.topology import build_stages
from contiguous_fuxian.sharedkv_v24.capture import ScoreCollector
from contiguous_fuxian.sharedkv_v24.restore_audit import ReadObserver
from contiguous_fuxian.sharedkv_v24.train import initialize,evaluate

torch.set_num_threads(1)

class Rotary(nn.Module):
    def forward(self,h,pos):
        d=4;f=torch.tensor([1.,.1],device=h.device)
        angle=pos.float()[...,None]*f
        angle=torch.cat((angle,angle),-1)
        return angle.cos().to(h.dtype),angle.sin().to(h.dtype)
class Attn(nn.Module):
    def __init__(self,d=4,hq=4,hk=2):
        super().__init__();width=d*hq
        self.config=types.SimpleNamespace(num_attention_heads=hq,num_key_value_heads=hk,hidden_size=width)
        self.head_dim=d
        self.q_proj=nn.Linear(width,hq*d);self.k_proj=nn.Linear(width,hk*d)
        self.v_proj=nn.Linear(width,hk*d);self.o_proj=nn.Linear(width,width,bias=False)
class Layer(nn.Module):
    def __init__(self):
        super().__init__();self.self_attn=Attn()
        self.input_layernorm=nn.LayerNorm(16);self.post_attention_layernorm=nn.LayerNorm(16)
        self.mlp=nn.Sequential(nn.Linear(16,24),nn.GELU(),nn.Linear(24,16))
class Tiny(nn.Module):
    def __init__(self,layers=4):
        super().__init__();self.model=nn.Module()
        self.model.embed_tokens=nn.Embedding(32,16);self.model.layers=nn.ModuleList([Layer() for _ in range(layers)])
        self.model.rotary_emb=Rotary();self.model.norm=nn.LayerNorm(16);self.lm_head=nn.Linear(16,32,bias=False)
        self.eval();self.requires_grad_(False)

def row(n=4,query=(2,3,4),split='fit'):
    q=list(query);plan=[[0,1,2] for _ in range(n)]
    labels={'a':[5],'b':[6,7],'c':[8,9,10]};lp={'a':[-2.],'b':[-2.,-3.],'c':[-3.,-2.,-1.]}
    return dict(format=FORMAT,split=split,task='toy',query_hash=digest_ids(q),query_token_ids=q,
        token_hash='prefix',prefix_tokens=4,geometry={'layers':n,'q_heads':4,'kv_heads':2,'head_dim':4},
        period=8 if n>4 else 4,keep_ratio=.25,model_identity={'weights':'toy'},teacher_sources=list(range(n)),
        label_order=list(labels),label_ids=labels,teacher_token_logprobs=lp,
        teacher_scores={k:sum(v)/len(v) for k,v in lp.items()},score_rule='mean_continuation_logprob',
        selection_by_layer=plan,selection_hash=json_digest(plan))

def fixture_model():
    torch.manual_seed(5);model=Tiny()
    raw={i:(torch.randn(4,2,4),torch.randn(4,2,4)) for i in range(4)}
    sources=[0,2,2,2]
    readers=nn.ModuleDict({str(i):TrainableReader(Reader(torch.eye(4).repeat(2,1,1),
        torch.eye(4).repeat(2,1,1),torch.zeros(2,4),torch.zeros(2,4)),ste=False) for i in [1,3]})
    return model,raw,sources,readers

@pytest.mark.parametrize('n',[4,8,28])
def test_record_roundtrip(tmp_path,n):
    r=row(n);validate(r);save_new(tmp_path/'toy'/(r['query_hash']+'.pt'),r)
    assert load(tmp_path,'toy','fit')[r['query_hash']]['label_ids']==r['label_ids']

@pytest.mark.parametrize('field,value',[
    ('gold','a'),('score_rule','sum'),('query_hash','bad'),('teacher_sources',[0,0,2,3]),('split','test')])
def test_invalid_record(field,value):
    r=row();r[field]=value
    with pytest.raises(ValueError):validate(r)

def test_score_normalization_rejected():
    r=row();r['teacher_scores']['b']=-5
    with pytest.raises(ValueError):validate(r)

def test_duplicate_write(tmp_path):
    r=row();p=tmp_path/'x.pt';save_new(p,r)
    with pytest.raises(FileExistsError):save_new(p,r)

def test_split_leak():
    r=row();v=copy.deepcopy(r);v['split']='validation'
    with pytest.raises(ValueError):assert_disjoint({r['query_hash']:r},{v['query_hash']:v})

@pytest.mark.parametrize('sources,period',[
    ([1,1,2,3],4),([0,2,1,3],4),([0,1,2,9],4),([0,1,2,3,4,1,6,7],4)])
def test_bad_sources(sources,period):
    with pytest.raises(ValueError):validate_sources(sources,period)

def test_unequal_plan():
    r=row();r['selection_by_layer'][1]=[0,1]
    with pytest.raises(ValueError):assert_plan(r,[0,2,2,2])

@pytest.mark.parametrize('protect,last_g',[(0,14),(4,16),(8,18)])
def test_budget_ladder(protect,last_g):
    r=row(28)
    for i in range(0,28,4):r['selection_by_layer'][i]=[0,1,2,3]
    r['selection_hash']=json_digest(r['selection_by_layer'])
    groups,stages=build_stages({r['query_hash']:r},protect_first=protect)
    assert stages[0]['sources'][25:28]==[26,26,26]
    assert stages[0]['unique_sources']==26
    assert stages[-1]['unique_sources']==last_g
    for s in stages:assert_plan(r,s['sources'])

def test_no_compatible_plan():
    r=row()
    r['selection_by_layer']=[[0],[1],[2],[3]]
    groups,stages=build_stages({r['query_hash']:r},protect_first=0)
    assert not stages

def test_unique_bank_alias():
    m,raw,ss,rr=fixture_model();r=row();bank=selected_bank(raw,r,ss,'cpu',torch.float32)
    assert len(bank)==2
    assert bank[ss[1]][0].data_ptr()==bank[ss[3]][0].data_ptr()

def test_logprob_length_and_final_token():
    m,raw,ss,rr=fixture_model();r=row();bank=selected_bank(raw,r,ss,'cpu',torch.float32)
    a=continuation_tokens(m,r,'c',ss,bank,rr,checkpoint_layers=False)
    r2=copy.deepcopy(r);r2['label_ids']['c'][-1]=11
    b=continuation_tokens(m,r2,'c',ss,bank,rr,checkpoint_layers=False)
    assert len(a)==3;torch.testing.assert_close(a[:-1],b[:-1]);assert float(a[-1].detach())!=float(b[-1].detach())

def test_single_vs_multitoken_mean():
    x=[torch.tensor([-2.]),torch.tensor([-1.,-2.,-3.])]
    torch.testing.assert_close(score_vectors(x),torch.tensor([-2.,-2.]))

def test_native_parity_self():
    m,raw,ss,rr=fixture_model();r=row();native=list(range(4));bank=selected_bank(raw,r,native,'cpu',torch.float32)
    with torch.no_grad():lp=[continuation_tokens(m,r,l,native,bank,{},checkpoint_layers=False) for l in r['label_order']]
    r['teacher_token_logprobs']={l:x.tolist() for l,x in zip(r['label_order'],lp)}
    r['teacher_scores']={l:float(x.mean()) for l,x in zip(r['label_order'],lp)}
    args=types.SimpleNamespace(temperature=1.,score_weight=1.,token_weight=.1,rank_weight=.25,margin_cap=.5,max_sequence=256)
    report=evaluate(m,{r['query_hash']:r},raw,ss,rr,args,native_parity=True)
    assert report['max_abs_mean_score_error']<1e-7

def test_grads_both_readers_backbone_frozen():
    m,raw,ss,rr=fixture_model();r=row();bank=selected_bank(raw,r,ss,'cpu',torch.float32)
    continuation_tokens(m,r,'b',ss,bank,rr).sum().backward()
    for reader in rr.values():
        assert sum(float(p.grad.abs().sum()) for p in reader.parameters() if p.grad is not None)>0
    assert all(p.grad is None for p in m.parameters())

@pytest.mark.parametrize('checkpoint_layers',[False,True])
def test_streaming_vjp_matches_full_graph(checkpoint_layers):
    m,raw,ss,rr=fixture_model();r=row();bank=selected_bank(raw,r,ss,'cpu',torch.float32)
    labels=r['label_order'];teacher=[torch.tensor(r['teacher_token_logprobs'][l]) for l in labels]
    compute=lambda l:continuation_tokens(m,r,l,ss,bank,rr,checkpoint_layers=checkpoint_layers)
    loss,_=objective([compute(l) for l in labels],teacher);loss.backward()
    expected=[p.grad.clone() for p in rr.parameters()]
    for p in rr.parameters():p.grad=None
    actual,_,_=streamed_backward(compute,labels,teacher,{})
    assert abs(actual-float(loss.detach()))<1e-6
    for p,e in zip(rr.parameters(),expected):torch.testing.assert_close(p.grad,e,atol=2e-6,rtol=2e-5)

def test_mse_shift_invariance_when_tokens_disabled():
    teacher=[torch.tensor([-2.]),torch.tensor([-3.])]
    student=[x+10 for x in teacher]
    loss,_=objective(student,teacher,token_weight=0)
    assert abs(float(loss))<1e-8

def test_rank_penalty_for_flip():
    t=[torch.tensor([-1.]),torch.tensor([-2.])];s=[torch.tensor([-3.]),torch.tensor([-1.])]
    _,parts=objective(s,t)
    assert parts['margin_loss']>0

def test_sequence_limit():
    m,raw,ss,rr=fixture_model();r=row();bank=selected_bank(raw,r,ss,'cpu',torch.float32)
    with pytest.raises(ValueError):continuation_tokens(m,r,'c',ss,bank,rr,max_sequence=3)

def test_initialize_refuses_reassignment():
    m,raw,ss,rr=fixture_model()
    initial={'format':'pans-readside-shared-prefix-v1','sources':[0,1,2,2],
             'readers':{'3':rr['3'].export().state_dict()}}
    with pytest.raises(ValueError):initialize(raw,[0,1,1,1],initial,row()['geometry'],.001,'cpu')

def test_end_to_end_synthetic_fit_export(tmp_path):
    m,raw,ss,unused=fixture_model();r=row();native=list(range(4))
    bank=selected_bank(raw,r,native,'cpu',torch.float32)
    with torch.no_grad():target=[continuation_tokens(m,r,l,native,bank,{},checkpoint_layers=False) for l in r['label_order']]
    r['teacher_token_logprobs']={l:t.tolist() for l,t in zip(r['label_order'],target)}
    r['teacher_scores']={l:sum(t.tolist())/len(t) for l,t in zip(r['label_order'],target)}
    save_new(tmp_path/'records/toy'/(r['query_hash']+'.pt'),r)
    ds=load(tmp_path/'records','toy','fit');_,stages=build_stages(ds,protect_first=0)
    ss=stages[0]['sources'];rr=initialize(raw,ss,None,r['geometry'],.01,'cpu')
    bank=selected_bank(raw,r,ss,'cpu',torch.float32);optim=torch.optim.Adam(rr.parameters(),lr=1e-4)
    for _ in range(2):
        optim.zero_grad(set_to_none=True)
        compute=lambda l:continuation_tokens(m,r,l,ss,bank,rr)
        streamed_backward(compute,r['label_order'],target,{})
        optim.step()
    exported={k:v.export().state_dict() for k,v in rr.items()}
    save_new(tmp_path/'checkpoint.pt',{'sources':ss,'readers':exported,'diagnostic_only':True})
    chk=torch.load(tmp_path/'checkpoint.pt',weights_only=True)
    assert len(set(chk['sources']))==2
    assert all(torch.isfinite(t).all() for v in chk['readers'].values() for t in v.values())
    assert all(p.grad is None for p in m.parameters())

def test_scorecollector_actual_kwargs(tmp_path):
    r=row();loader=types.SimpleNamespace(_info=types.SimpleNamespace(task='toy',token_hash='prefix'),
        prefix_tokens=4,keep_ratio=.25,_selector_index=types.SimpleNamespace(payload={'model_identity':r['model_identity']}),
        _selected_by_layer=r['selection_by_layer'])
    runtime=types.SimpleNamespace(mode='identity',backend='mock',layout='hybrid',
        _prepare=lambda *a:(r['geometry'],list(range(4)),{},set()))
    def greedy(**kw):
        kw['label_token_logprobs_out'].update(copy.deepcopy(r['teacher_token_logprobs']));return ('x',1.,1.,{})
    runner=types.SimpleNamespace(greedy_flexgen_completion=greedy)
    ScoreCollector(tmp_path,'fit').install(runtime,runner)
    out={};res=runner.greedy_flexgen_completion(model=object(),loader=loader,max_tokens=1,
        impress_selection_period_size=4,query_token_ids=r['query_token_ids'],label_continuations=r['label_ids'],label_token_logprobs_out=out)
    assert res[0]=='x';loaded=load(tmp_path,'toy','fit')[r['query_hash']]
    assert loaded['teacher_scores']['b']==-2.5
    assert loaded['teacher_token_logprobs']==out


def test_observer_restores_and_separates_paths():
    class Token:
        def __init__(self):self.device='disk';self.path='a.npy';self.dtype=np.float16;self.shape=(16,2,4)
        def to(self,device):self.device=device
    class Prefix:
        def _chunk_source_readonly(self,chunk,want_key):return 'original'
    module=types.SimpleNamespace(TokenCacheDisk=Token,PrefixKVLayer=Prefix)
    token=Token();chunk=types.SimpleNamespace(token=token);orig=Token.to
    observer=ReadObserver(module)
    with observer:
        assert Prefix()._chunk_source_readonly(chunk,True)=='original'
        token.to('cpu')
    assert Token.to is orig
    report=observer.report();assert report['files_seen_in_both_paths']==1
    assert report['declared_overlap_file_bytes']==256
    assert report['formal_timing_eligible'] is False


def test_observer_restore_after_exception():
    class Token:
        def to(self,device):pass
    class Prefix:
        def _chunk_source_readonly(self,chunk,want_key):pass
    m=types.SimpleNamespace(TokenCacheDisk=Token,PrefixKVLayer=Prefix);old=Token.to
    with pytest.raises(RuntimeError):
        with ReadObserver(m):raise RuntimeError('test')
    assert old is Token.to


def quality_module():
    path=Path(__file__).resolve().parents[1]/'scripts/audit_quality_v24.py'
    spec=importlib.util.spec_from_file_location('qa',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.mark.parametrize('n,expected',[(175,False),(500,True)])
def test_no_flip_noninferiority_not_degenerate(n,expected):
    m=quality_module();a={i:{'correct':bool(i%2)} for i in range(n)}
    r=m.paired_report(a,copy.deepcopy(a));assert r['conservative_bound_passes'] is expected

def test_current_four_vs_three_not_certified():
    m=quality_module();a={i:{'correct':i<343} for i in range(500)};b=copy.deepcopy(a)
    for i in range(4):b[i]['correct']=False
    for i in range(343,346):b[i]['correct']=True
    r=m.paired_report(a,b)
    assert r['net_delta_pp']==-.2;assert not r['conservative_bound_passes']

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')
def test_cuda_reader_gradient():
    m,raw,ss,rr=fixture_model();m=m.cuda().to(torch.bfloat16);rr=rr.cuda()
    raw={k:tuple(x.cuda().to(torch.bfloat16) for x in v) for k,v in raw.items()};r=row()
    bank=selected_bank(raw,r,ss,'cuda',torch.bfloat16)
    continuation_tokens(m,r,'b',ss,bank,rr).mean().backward()
    assert all(p.grad is None for p in m.parameters())
    assert any(p.grad is not None for p in rr.parameters())

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')
def test_cuda_source_alias():
    _,raw,ss,_=fixture_model();r=row();bank=selected_bank(raw,r,ss,'cuda',torch.bfloat16)
    assert bank[ss[1]][0].data_ptr()==bank[ss[3]][0].data_ptr()


def test_margin_sufficient_condition():
    torch.manual_seed(12)
    teacher=torch.randn(100,6);student=teacher+.1*torch.randn(100,6)
    e=((student-student.mean(-1,keepdim=True))-(teacher-teacher.mean(-1,keepdim=True))).abs().amax(-1)
    top=teacher.sort(-1,descending=True).values
    certified=top[:,0]-top[:,1]>2*e
    assert torch.equal(student.argmax(-1)[certified],teacher.argmax(-1)[certified])


def test_plan_design_does_not_read_labels():
    r=row(28);base=build_stages({r['query_hash']:r},protect_first=4)[1]
    altered=copy.deepcopy(r);altered['teacher_scores']={k:999-i for i,k in enumerate(r['label_order'])}
    other=build_stages({altered['query_hash']:altered},protect_first=4)[1]
    assert [x['sources'] for x in base]==[x['sources'] for x in other]
