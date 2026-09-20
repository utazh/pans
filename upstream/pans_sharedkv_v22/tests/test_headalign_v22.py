import copy,itertools,json,types
from pathlib import Path
import pytest
import torch
from contiguous_fuxian.sharedkv.core import Reader,attention_with_lse,segmented_attention,apply_rope
from contiguous_fuxian.sharedkv_v21 import ops as old_ops
from contiguous_fuxian.sharedkv_v21.refine import distribution
from contiguous_fuxian.sharedkv_v22.matching import (validate_permutation,choose_assignment,split_fit,
    normalized_maps,aligned_fingerprint,FORMAT)
from contiguous_fuxian.sharedkv_v22.ops import prepare_route,mapped_attention
from contiguous_fuxian.sharedkv_v22.calibrate import fit_assignment
from contiguous_fuxian.sharedkv_v22.budget import budget_frontier

torch.set_num_threads(2)


def data(h=4,g=2,d=8,t=3,n=11,nt=5,dtype=torch.float64,device='cpu'):
    torch.manual_seed(31)
    r=lambda *s:torch.randn(*s,dtype=dtype,device=device)
    q,pk,pv,tk,tv=r(1,t,h*g,d),r(1,n,h,d),r(1,n,h,d),r(1,nt,h,d),r(1,nt,h,d)
    reader=Reader(r(h,d,d)/d**.5,r(h,d,d)/d**.5,r(h,d)*.3,r(h,d)*.3)
    return q,pk,pv,tk,tv,reader


def explicit(q,pk,pv,tk,tv,r,p):
    k=torch.einsum('bnhd,hde->bnhe',pk[:,:,list(p)],r.A)+r.bk
    v=torch.einsum('bnhd,hde->bnhe',pv[:,:,list(p)],r.B)+r.bv
    return attention_with_lse(q,torch.cat((k,tk),1),torch.cat((v,tv),1),causal=True)[0]


@pytest.mark.parametrize('p',[[0,1,2,3],[1,0,3,2],[2,0,3,1],[3,2,1,0]])
@pytest.mark.parametrize('packed',[False,True])
def test_exact_affine_joint_attention(p,packed):
    q,pk,pv,tk,tv,r=data()
    route=prepare_route(p,q_heads=q.shape[2],device='cpu')
    out=mapped_attention(q,pk,pv,tk,tv,reader=r,route=route,packed=old_ops.pack(r) if packed else None)
    assert torch.allclose(out,explicit(q,pk,pv,tk,tv,r,p),atol=2e-10,rtol=2e-10)


def test_shared_prefix_objects_unchanged_no_head_expansion():
    q,pk,pv,tk,tv,r=data();kcopy=pk.clone();vcopy=pv.clone();seen=[]
    def spy(qi,ki,vi,**kw):
        seen.append((ki,vi));return attention_with_lse(qi,ki,vi,**kw)
    mapped_attention(q,pk,pv,tk,tv,reader=r,route=prepare_route([2,0,3,1],q_heads=8,device='cpu'),attention_fn=spy)
    assert len(seen)==2 and seen[0][0] is pk and seen[0][1] is pv
    assert seen[1][0] is tk and seen[1][1] is tv
    assert torch.equal(pk,kcopy) and torch.equal(pv,vcopy)


@pytest.mark.parametrize('t,nt',[(1,7),(3,3),(5,8)])
def test_private_tail_native_causal_and_decode(t,nt):
    q,pk,pv,tk,tv,r=data(t=t,nt=nt)
    p=[3,0,1,2]
    out=mapped_attention(q,pk,pv,tk,tv,reader=r,route=prepare_route(p,q_heads=8,device='cpu'))
    assert torch.allclose(out,explicit(q,pk,pv,tk,tv,r,p),atol=1e-10,rtol=1e-10)


def test_rope_remains_in_original_token_positions():
    q,pk,pv,tk,tv,r=data()
    pos=torch.arange(q.shape[1],dtype=q.dtype)[None,:,None]
    freq=torch.tensor([.1,.2,.3,.4],dtype=q.dtype).repeat(2)[None,None]
    q=apply_rope(q,(pos*freq).cos(),(pos*freq).sin())
    p=[1,3,0,2]
    result=mapped_attention(q,pk,pv,tk,tv,reader=r,route=prepare_route(p,q_heads=8,device='cpu'))
    assert torch.allclose(result,explicit(q,pk,pv,tk,tv,r,p),atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('p',[[0,0,2,3],[0,1,2],[True,1,2,3],[-1,1,2,3],[0,1,2,4]])
def test_bad_maps_rejected(p):
    with pytest.raises(ValueError):validate_permutation(p,4)


def test_exact_assignment_and_identity_tie():
    c=torch.ones(4,4)
    assert choose_assignment(c)['permutation']==[0,1,2,3]
    wanted=[2,0,3,1]
    c[torch.arange(4),wanted]=.1
    r=choose_assignment(c)
    assert r['permutation']==wanted and r['matched_cost']<r['identity_cost']


@pytest.mark.parametrize('bad',[torch.ones(2,3),torch.tensor([[float('nan')]]),torch.tensor([[-2.]])])
def test_bad_cost_rejected(bad):
    with pytest.raises(ValueError):choose_assignment(bad)


def test_internal_split_by_query_no_overlap():
    rows=[({'query_hash':str(i)}, {}) for i in range(64)]
    a,b=split_fit(rows)
    assert len(a)==48 and len(b)==16
    assert not {x[0]['query_hash'] for x in a}&{x[0]['query_hash'] for x in b}
    assert split_fit(rows)==(a,b)


def test_permutation_fingerprint_changes_and_identity_legacy_safe():
    *_,r=data(dtype=torch.float32)
    readers={3:r}
    assert aligned_fingerprint(readers,{3:(0,1,2,3)})==aligned_fingerprint(readers,{})
    assert aligned_fingerprint(readers,{3:(1,0,2,3)})!=aligned_fingerprint(readers,{})
    with pytest.raises(ValueError):normalized_maps({'7':[0,1,2,3]},[3],4)


def synthetic_rows(ks,vs,p,requests=12):
    h,d=ks.shape[1:];g=2;n=ks.shape[0]
    kt,vt=ks[:,p].clone(),vs[:,p].clone()
    rows=[]
    for i in range(requests):
        q=torch.randn(5,h*g,d)*.6
        tail=torch.randn_like(q)*.2
        lse=torch.zeros(5,h*g)
        out,_,_=distribution(q,kt,vt,tail,lse)
        record={'q':q,'tail_output':tail,'tail_lse':lse,'attention_output':out,
                'selected_tokens':torch.arange(n),'query_positions':torch.arange(5)}
        row={'query_hash':str(i),'layers':{3:record}}
        rows.append((copy.deepcopy(row),copy.deepcopy(row)))
    return rows,kt,vt


def test_fitting_recovers_planted_head_permutation():
    torch.manual_seed(10)
    ks=torch.randn(96,4,4);vs=torch.randn_like(ks)
    p=[2,0,3,1]
    rows,kt,vt=synthetic_rows(ks,vs,p)
    report,_=fit_assignment(rows,3,ks,vs,kt,vt,ridge=1e-5,device='cpu')
    assert report['permutation']==p
    assert report['matched_cost']<report['identity_cost']*.01
    # This is deliberately synthetic and must not be described as a Qwen result.


def test_budget_counts_readers_and_does_not_claim_h2d_or_override():
    r=budget_frontier(consumers=3)
    assert r['baseline_payload_bytes']==70647808
    assert r['same_retention_shared_payload_bytes']==63078400  # full-block uniform model; not observed tail-aware bytes
    assert r['reader_and_route_bytes']==3*(264192+133120+448)
    assert not r['automatic_runtime_retention_change']
    s=budget_frontier(consumers=0)
    assert s['net_persistent_saving_bytes']==0 and s['iso_payload_plus_readers_max_uniform_blocks']==77


def test_runtime_compatibility_view_does_not_edit_bundle(tmp_path,monkeypatch):
    from contiguous_fuxian.sharedkv_v22.runtime import Runtime
    from contiguous_fuxian.sharedkv_v21 import runtime as parent
    from contiguous_fuxian.sharedkv.io import FORMAT as OLD_FORMAT,load_safe
    *_,r=data(dtype=torch.float32)
    bundle={'format':FORMAT,'sources':[0,1,2,2],'geometry':{'kv_heads':4,'q_heads':8},
            'readers':{'3':r.state_dict()},'head_maps':{'3':[2,0,3,1]}}
    torch.save(bundle,tmp_path/'trec.pt');before=(tmp_path/'trec.pt').read_bytes()
    def fake_prepare(self,model,loader,period):
        if 'trec' not in self.prepared:
            b=load_safe(self.adapter_dir/'trec.pt');assert b['format']==OLD_FORMAT
            rr={3:Reader.from_state_dict(b['readers']['3'])}
            self.prepared['trec']=({'q_heads':8},b['sources'],rr,set())
            self.metadata['trec']={}
        return self.prepared['trec']
    monkeypatch.setattr(parent.Runtime,'_prepare',fake_prepare)
    def fake_init(self,**kw):
        self.prepared={}; self.metadata={}; self.routes={}
        self.mode=kw['mode'];self.adapter_dir=Path(kw['adapter_dir'])
    monkeypatch.setattr(parent.Runtime,'__init__',fake_init)
    rt=Runtime(mode='shared',backend='reference',adapter_dir=tmp_path,allow_candidate=True)
    model=types.SimpleNamespace(parameters=lambda:iter([torch.zeros(1)]))
    loader=types.SimpleNamespace(_info=types.SimpleNamespace(task='trec'))
    rr=rt._prepare(model,loader,8)[2]
    assert rt.head_maps['trec'][3]==(2,0,3,1)
    assert rt.metadata['trec']['rollout_reader_fingerprint']==aligned_fingerprint(rr,rt.head_maps['trec'])
    assert (tmp_path/'trec.pt').read_bytes()==before
    assert rt.routes['trec'][id(rr[3])].resident_bytes==2*8*8
    assert rt._prepare(model,loader,8)[2] is rr
    rt._view.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_flash2_permutation_parity():
    pytest.importorskip('flash_attn')
    q,pk,pv,tk,tv,r=data(d=32,g=7,dtype=torch.float16,device='cuda')
    route=prepare_route([2,0,3,1],q_heads=28,device='cuda')
    ref=mapped_attention(q,pk,pv,tk,tv,reader=r,route=route,backend='reference')
    result=mapped_attention(q,pk,pv,tk,tv,reader=r,route=route,backend='flash2')
    assert torch.allclose(result,ref,atol=.03,rtol=.03)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_vllm2_permutation_parity():
    pytest.importorskip('vllm.vllm_flash_attn')
    q,pk,pv,tk,tv,r=data(d=32,g=7,dtype=torch.bfloat16,device='cuda')
    route=prepare_route([1,3,0,2],q_heads=28,device='cuda')
    result=mapped_attention(q,pk,pv,tk,tv,reader=r,route=route,backend='vllm2')
    ref=mapped_attention(q,pk,pv,tk,tv,reader=r,route=route,backend='reference')
    assert torch.allclose(result,ref,atol=.08,rtol=.08)


def test_complete_synthetic_compile_and_gradient_step(tmp_path):
    from contiguous_fuxian.sharedkv_v22.calibrate import main
    from contiguous_fuxian.sharedkv_v21.traces import TRACE_FORMAT,fingerprint
    from contiguous_fuxian.sharedkv.io import load_safe
    torch.manual_seed(4)
    n,h,d=48,4,4;p=[2,0,3,1]
    ks=torch.randn(n,h,d).to(torch.bfloat16).float()
    vs=torch.randn_like(ks).to(torch.bfloat16).float()
    kt,vt=ks[:,p],vs[:,p]
    store=tmp_path/'store'/'toy';store.mkdir(parents=True)
    meta={'token_hash':'synthetic-prefix','prefix_tokens':n,'kv_heads':h,'head_dim':d,
          'layers':4,'dtype':'bfloat16','layout':'token,kv_head,head_dim'}
    (store/'metadata.json').write_text(json.dumps(meta))
    for lid,k,v in [(2,ks,vs),(3,kt,vt)]:
        for name,x in [('key',k),('value',v)]:
            x.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tofile(store/f'layer_{lid:02d}_{name}.bf16')
    geom={'layers':4,'q_heads':8,'kv_heads':4,'head_dim':4};identity={'synthetic':'not-real-model'}
    header={'format':TRACE_FORMAT,'task':'toy','token_hash':'synthetic-prefix','prefix_tokens':n,
            'geometry':geom,'model_identity':identity,'period':4,'keep_ratio':.25,
            'rollout_sources':list(range(4)),'rollout_reader_fingerprint':fingerprint({}),
            'selection_by_layer':[list(range(n)) for _ in range(4)]}
    for split,number in [('fit',8),('val',4)]:
        directory=tmp_path/split/'toy';directory.mkdir(parents=True)
        rows,_,_=synthetic_rows(ks,vs,p,requests=number)
        for i,(r,_) in enumerate(rows):
            l=r['layers'][3];o=l['attention_output'];z=torch.zeros(o.shape[0],32)
            l.update(hidden_in=z,pre_mlp=o.flatten(1),hidden_out=o.flatten(1))
            torch.save({**header,'query_hash':f'{split}-{i}','layers':{3:l}},directory/f'{split}-{i}.pt')
    weightdir=tmp_path/'fit'/'weights';weightdir.mkdir()
    torch.save({'o_proj':torch.eye(32),'model_identity':identity},weightdir/'layer_03.pt')
    output=tmp_path/'out.pt'
    main(['--store-root',str(tmp_path/'store'),'--task','toy',
          '--teacher-fit',str(tmp_path/'fit'),'--student-fit',str(tmp_path/'fit'),
          '--teacher-val',str(tmp_path/'val'),'--student-val',str(tmp_path/'val'),
          '--source','2','--target','3','--steps','1','--eval-every','1','--bf16-ste',
          '--output',str(output),'--ridge','0.00001'])
    b=load_safe(output)
    assert b['format']==FORMAT and b['head_maps']['3']==p
    assert b['diagnostic_only'] and not b['end_to_end_validated']
    assert set(b['calibration_query_hashes'])=={f'fit-{i}' for i in range(8)}|{f'val-{i}' for i in range(4)}
    assert len(json.loads(output.with_suffix('.report.json').read_text())['history'])==2


def test_data_gate_rejects_holdout_fit_overlap():
    from contiguous_fuxian.sharedkv_v21.traces import assert_splits
    with pytest.raises(ValueError):assert_splits({'same':1},{'same':2})
