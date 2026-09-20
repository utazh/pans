import copy,json,math
from pathlib import Path
import pytest
import torch
from contiguous_fuxian.sharedkv.core import Reader,segmented_attention
from contiguous_fuxian.sharedkv.io import save_new
from contiguous_fuxian.sharedkv_v21.ops import pack,prepare_query,finish,packed_attention,native_attention,dispatch
from contiguous_fuxian.sharedkv_v21.refine import distribution,TrainableReader,objective,fit_value_asymmetric,main as refine_main
from contiguous_fuxian.sharedkv_v21.audit import logit_span_bound,paired_quality,audit_traces
from contiguous_fuxian.sharedkv_v21.traces import TRACE_FORMAT,paired,assert_splits,fingerprint,load_traces


def tensors(dtype=torch.float64,t=3,tail=5):
    torch.manual_seed(8);h,hq,d,n=2,4,4,9
    def r(*shape):return torch.randn(*shape,dtype=dtype)*.3
    q=r(1,t,hq,d);pk=r(1,n,h,d);pv=r(1,n,h,d);tk=r(1,tail,h,d);tv=r(1,tail,h,d)
    eye=torch.eye(d,dtype=dtype).repeat(h,1,1)
    reader=Reader(eye+r(h,d,d),eye+r(h,d,d),r(h,d),r(h,d))
    return q,pk,pv,tk,tv,reader


@pytest.mark.parametrize("t,tail",[(1,1),(1,5),(3,3),(3,7)])
def test_hybrid_native_joint_mask(t,tail):
    q,pk,pv,tk,tv,_=tensors(t=t,tail=tail)
    a=segmented_attention(q,pk,pv,tk,tv)
    b=native_attention(q,pk,pv,tk,tv)
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64,torch.bfloat16])
def test_packed_reader_equivalence(dtype):
    q,pk,pv,tk,tv,r=tensors(dtype=dtype)
    a=segmented_attention(q,pk,pv,tk,tv,reader=r)
    b=packed_attention(q,pk,pv,tk,tv,pack(r))
    tol=.015 if dtype==torch.bfloat16 else (2e-6 if dtype==torch.float32 else 1e-12)
    torch.testing.assert_close(a,b,atol=tol,rtol=tol)


def test_differentiable_distribution_equivalent_to_segmented():
    from contiguous_fuxian.sharedkv.core import attention_with_lse
    q,pk,pv,tk,tv,r=tensors()
    u,z=attention_with_lse(q,tk,tv,causal=True)
    o,logp,beta=distribution(q[0],pk[0],pv[0],u[0],z[0].T,r.A,r.B,r.bk,r.bv)
    ref=segmented_attention(q,pk,pv,tk,tv,reader=r)
    torch.testing.assert_close(o,ref[0],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(logp.exp().sum(-1),torch.ones_like(logp[...,0]))
    torch.testing.assert_close(beta,1-logp.exp()[...,-1].T)


def test_key_offset_changes_joint_mass_not_prefix_distribution():
    q,pk,pv,tk,tv,r=tensors()
    tail_out=torch.zeros_like(q[0]);tail_lse=torch.zeros(q.shape[1],q.shape[2],dtype=q.dtype)
    _,p0,b0=distribution(q[0],pk[0],pv[0],tail_out,tail_lse,r.A,r.B,torch.zeros_like(r.bk),r.bv)
    _,p1,b1=distribution(q[0],pk[0],pv[0],tail_out,tail_lse,r.A,r.B,10*r.bk,r.bv)
    torch.testing.assert_close(p0[...,:-1].softmax(-1),p1[...,:-1].softmax(-1))
    assert not torch.allclose(b0,b1)


def test_reader_gradient_nonzero_and_no_backbone_parameter():
    q,pk,pv,tk,tv,r=tensors(dtype=torch.float32)
    module=TrainableReader(r,ste=True)
    out,lp,_=distribution(q[0],pk[0],pv[0],torch.zeros_like(q[0]),torch.zeros(q.shape[1],q.shape[2]),*module.tensors())
    (out.square().mean()+lp.square().mean()).backward()
    assert set(dict(module.named_parameters()))=={"A","B","bk","bv"}
    for p in module.parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0


def test_simple_reader_training_reduces_synthetic_loss():
    q,pk,pv,tk,tv,r=tensors(dtype=torch.float32)
    uv=torch.zeros_like(q[0]);z=torch.zeros(q.shape[1],q.shape[2])
    with torch.no_grad():target,tlog,_=distribution(q[0],pk[0],pv[0],uv,z,r.A,r.B,r.bk,r.bv)
    eye=torch.eye(4).repeat(2,1,1);m=TrainableReader(Reader(eye,eye,torch.zeros(2,4),torch.zeros(2,4)))
    opt=torch.optim.Adam(m.parameters(),lr=.02)
    losses=[]
    for _ in range(40):
        out,logp,_=distribution(q[0],pk[0],pv[0],uv,z,*m.tensors())
        at,kl,_=objective(out,logp,target,tlog);loss=at+kl
        opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
    assert losses[-1]<losses[0]*.3


def test_postprojection_loss_accounts_for_upstream_residual():
    o=torch.ones(2,2,2);lp=torch.zeros(2,2,3).log_softmax(-1)
    h=torch.ones(2,4);w=torch.eye(4)
    a,kl,post=objective(o,lp,o,lp,student_hidden=h+1,teacher_hidden=h,
                        teacher_pre_mlp=h+o.flatten(1),o_weight=w)
    assert a==0 and abs(float(kl))<1e-9 and post>0


def test_logit_span_lower_bound_and_vacuous_case():
    torch.manual_seed(1);x=torch.randn(30,4,dtype=torch.float64);coef=torch.randn(7,4,dtype=x.dtype)
    good=coef@x.T+3
    a=logit_span_bound(x,good)
    assert a["rmse"]<1e-12 and not a["potentially_vacuous"]
    b=logit_span_bound(x,torch.randn_like(good))
    assert b["relative_to_centered_logit_energy"]>.4
    assert logit_span_bound(x[:3],good[:,:3])["potentially_vacuous"]


def test_paired_quality_reports_discordances_not_just_means():
    b=[{"task":"t","query_hash":str(i),"correct":int(i<6)} for i in range(10)]
    s=copy.deepcopy(b);s[0]["correct"]=0;s[8]["correct"]=1
    r=paired_quality(b,s,bootstrap=100)
    assert r["delta_pp"]==0 and r["baseline_right_student_wrong"]==1 and r["baseline_wrong_student_right"]==1
    with pytest.raises(ValueError):paired_quality(b,s[:-1])
    with pytest.raises(ValueError):paired_quality(b+[b[0]],s)


def make_header(hash_):
    return {"format":TRACE_FORMAT,"task":"toy","query_hash":hash_,"token_hash":"prefix",
        "prefix_tokens":9,"geometry":{"layers":4,"q_heads":4,"kv_heads":2,"head_dim":4},
        "model_identity":{"fixture":1},"period":4,"keep_ratio":.25,
        "rollout_sources":[0,1,2,3],"rollout_reader_fingerprint":fingerprint({}),
        "selection_by_layer":[list(range(9)) for _ in range(4)]}


def make_record(hash_, *, student_delta=0.):
    from contiguous_fuxian.sharedkv.core import attention_with_lse
    q,pk,pv,tk,tv,r=tensors(dtype=torch.float32)
    u,z=attention_with_lse(q,tk,tv,causal=True)
    out,_,_=distribution(q[0],pk[0],pv[0],u[0],z[0].T)
    h=torch.zeros(q.shape[1],16)
    dat={"q":q[0]+student_delta,"tail_output":u[0],"tail_lse":z[0].T,
         "query_positions":torch.arange(q.shape[1]),"selected_tokens":torch.arange(9),
         "attention_output":out,"hidden_in":h,"pre_mlp":h+out.flatten(1),"hidden_out":h+out.flatten(1)}
    return {**make_header(hash_),"layers":{i:copy.deepcopy(dat) for i in range(4)}}


def test_trace_pair_identity_leakage_and_drift_checks():
    t={"a":make_record("a")};s={"a":make_record("a",student_delta=.01)}
    paired(t,s,3);r=audit_traces(t,s)
    assert r["3"]["q"]["mean"]>0
    with pytest.raises(ValueError):assert_splits(t,s)
    s["a"]["layers"][3]["selected_tokens"]=torch.arange(8)
    with pytest.raises(ValueError):paired(t,s,3)
    paired(t,s,3,require_same_ids=False)


def test_asymmetric_value_fit_uses_student_design():
    r=make_record("a");s=copy.deepcopy(r)
    q,ks,vs,tk,tv,reader=tensors(dtype=torch.float32)
    s["layers"][3]["q"]*=1.5
    eye=torch.eye(4).repeat(2,1,1)
    initial=Reader(eye,eye,torch.zeros(2,4),torch.zeros(2,4))
    fitted=fit_value_asymmetric([(r,s)],3,ks[0],vs[0],ks[0],vs[0],initial,ridge=1e-4,device="cpu")
    assert not torch.equal(initial.B,fitted.B)


def test_compiler_end_to_end_synthetic(tmp_path):
    root=tmp_path
    q,pk,pv,_,_,_=tensors(dtype=torch.float32)
    store=root/"store/toy";store.mkdir(parents=True)
    metadata={"dtype":"bfloat16","layout":"token,kv_head,head_dim","prefix_tokens":9,
              "kv_heads":2,"head_dim":4,"layers":4,"token_hash":"prefix"}
    (store/"metadata.json").write_text(json.dumps(metadata))
    for l in range(4):
        for name,x in (("key",pk), ("value",pv)):
            (store/f"layer_{l:02d}_{name}.bf16").write_bytes(x[0].to(torch.bfloat16).view(torch.uint8).numpy().tobytes())
    for split in ("fit","val"):
        r=make_record(split)
        save_new(root/split/"toy"/f"{split}.pt",r)
        save_new(root/split/"weights/layer_03.pt",{"o_proj":torch.eye(16),"model_identity":{"fixture":1}})
    output=root/"out/toy.pt"
    argv=["--store-root",str(root/"store"),"--task","toy","--teacher-fit",str(root/"fit"),
          "--teacher-val",str(root/"val"),"--student-fit",str(root/"fit"),"--student-val",str(root/"val"),
          "--source","2","--target","3","--steps","2","--eval-every","1","--output",str(output)]
    refine_main(argv)
    result=torch.load(output,weights_only=True)
    assert result["sources"]==[0,1,2,2]
    assert result["deployment_status"]=="diagnostic_candidate_not_accepted"
    assert result["end_to_end_validated"] is False
    assert set(result["calibration_query_hashes"])=={"fit","val"}
    with pytest.raises(FileExistsError):refine_main(argv)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="No CUDA available")
def test_gpu_flash2_parity():
    pytest.importorskip("flash_attn")
    q,pk,pv,tk,tv,r=tensors(dtype=torch.float32)
    # Real supported FA dimension; source geometry is separately tested on CPU.
    h,hq,d=2,4,64
    q=torch.randn(1,3,hq,d,device="cuda",dtype=torch.bfloat16)*.2
    pk=torch.randn(1,9,h,d,device="cuda",dtype=torch.bfloat16)*.2;pv=torch.randn_like(pk)
    tk=torch.randn(1,5,h,d,device="cuda",dtype=torch.bfloat16)*.2;tv=torch.randn_like(tk)
    eye=torch.eye(d,device="cuda",dtype=torch.bfloat16).repeat(h,1,1)
    r=Reader(eye,eye,torch.zeros(h,d,device="cuda",dtype=torch.bfloat16),torch.zeros(h,d,device="cuda",dtype=torch.bfloat16))
    a=segmented_attention(q,pk,pv,tk,tv,reader=r,backend="flash2")
    b=packed_attention(q,pk,pv,tk,tv,pack(r),backend="flash2")
    torch.testing.assert_close(a,b,atol=.01,rtol=.01)


def test_mock_decoder_physically_shares_source_and_captures(tmp_path,monkeypatch):
    import types,sys
    from torch import nn
    from contiguous_fuxian.sharedkv import runtime as v1
    from contiguous_fuxian.sharedkv_v21 import runtime as v2
    from contiguous_fuxian.sharedkv_v21.traces import TraceSink
    class Attn(nn.Module):
        def __init__(self):
            super().__init__();self.q_proj=nn.Linear(16,16);self.k_proj=nn.Linear(16,8)
            self.v_proj=nn.Linear(16,8);self.o_proj=nn.Linear(16,16,bias=False)
    class Layer(nn.Module):
        def __init__(self):
            super().__init__();self.input_layernorm=nn.Identity();self.post_attention_layernorm=nn.Identity()
            self.self_attn=Attn();self.mlp=nn.Linear(16,16)
    class Core(nn.Module):
        def __init__(self):
            super().__init__();self.embed_tokens=nn.Embedding(20,16);self.layers=nn.ModuleList([Layer() for _ in range(4)])
            self.norm=nn.Identity()
        def rotary_emb(self,h,pos):return torch.ones(1,h.shape[1],4),torch.zeros(1,h.shape[1],4)
    class Cache:
        def __init__(self):self.data={}
        def get_seq_length(self,l):return self.data[l][0].shape[2] if l in self.data else 0
        def update(self,k,v,l,extra):
            if l in self.data:k,v=torch.cat((self.data[l][0],k),2),torch.cat((self.data[l][1],v),2)
            self.data[l]=(k,v);return k,v
    class Loader:
        def __init__(self):self._selected_by_layer=[[] for _ in range(4)];self._loaded={};self.calls=[]
        def schedule(self,l):pass
        def resolve(self,l):
            self.calls.append(l);return torch.randn(9,2,4)*.2,torch.randn(9,2,4)*.2
        def prefetch_selector_keys(self,l):pass
        def configure_impress_layer(self,layer,selected_tokens,prefetch_priority_tokens):self._selected_by_layer[layer]=selected_tokens
    model=types.SimpleNamespace(model=Core(),lm_head=nn.Linear(16,20))
    loader=Loader();g=make_header("x")["geometry"]
    eye=torch.eye(4).repeat(2,1,1);reader=Reader(eye,eye,torch.zeros(2,4),torch.zeros(2,4))
    state=v1.RequestState([0,1,2,2],{3:reader},"reference",geometry=g)
    def selector(**kw):
        for lid in range(4):kw["loader"]._selected_by_layer[lid]=list(range(9))
        return 4
    fake=types.ModuleType("contiguous_fuxian.flexgen_qwen_reprefill");fake.configure_online_layer_selection=selector
    monkeypatch.setitem(sys.modules,"contiguous_fuxian.flexgen_qwen_reprefill",fake)
    sink=TraceSink(tmp_path,"toy","x",make_header("x"),layers=range(4),max_samples=2)
    t1=v1._ACTIVE.set(state);t2=v2._SETTINGS.set({"packed":{3:pack(reader)},"packed_ops":True,
        "layout":"hybrid","nvtx":False,"sink":sink,"replay":None,"selector_calls":0,"shadow_buffers":{}})
    try:
        cache=Cache()
        with torch.inference_mode():
            out=v2.decoder_logits(model=model,input_ids=torch.tensor([[1,2,3]]),cache=cache,
                position_start=9,loader=loader,period_size=4,subperiod_size=2,impress_selection_period_size=4)
        assert out.shape==(1,3,20)
        assert loader.calls==[0,1,2] and state.resolve_calls==3
        assert state.resolve(loader,2,dtype=torch.float32,device="cpu")[0].data_ptr()==state.resolve(loader,3,dtype=torch.float32,device="cpu")[0].data_ptr()
        assert all(k.shape[2]==3 for k,v in cache.data.values()) # private tail only
        sink.save();loaded=load_traces(tmp_path,"toy")
        assert len(loaded["x"]["layers"])==4
        assert (tmp_path/"weights/layer_03.pt").exists()
    finally:v2._SETTINGS.reset(t2);v1._ACTIVE.reset(t1)
