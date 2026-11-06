import importlib.util
import types
import sys
import torch
import pytest
from contiguous_fuxian.sharedkv.core import (Reader, attention_with_lse, segmented_attention,
                                           rotate_half, query_transform)
from contiguous_fuxian.sharedkv.fit import (commuting_rope_fit,fit_key_reader,
                                          weighted_reduced_rank,fit_value_from_attention)
from contiguous_fuxian.sharedkv.io import validate_sources
from contiguous_fuxian.sharedkv.runtime import RequestState, decoder_logits, _ACTIVE

torch.set_num_threads(2)


def tensors(dtype=torch.float64,device="cpu",b=1,t=3,n=11,h=2,g=3,d=8,tail=5):
    torch.manual_seed(42)
    def r(*shape):return torch.randn(*shape,dtype=dtype,device=device)
    return r(b,t,h*g,d),r(b,n,h,d),r(b,n,h,d),r(b,tail,h,d),r(b,tail,h,d)


def dense_attn(q,k,v,causal=False):
    g=q.shape[2]//k.shape[2]
    k=k.repeat_interleave(g,dim=2);v=v.repeat_interleave(g,dim=2)
    s=torch.einsum("bthd,bnhd->bhtn",q,k)*q.shape[-1]**-0.5
    if causal:
        tq,n=q.shape[1],k.shape[1]
        mask=torch.arange(n)[None,:]<=torch.arange(n-tq,n)[:,None]
        s=s.masked_fill(~mask[None,None],-torch.inf)
    return torch.einsum("bhtn,bnhd->bthd",s.softmax(-1),v),s.logsumexp(-1)


@pytest.mark.parametrize("causal",[False,True])
@pytest.mark.parametrize("key_chunk",[2,7,32])
def test_streaming_attention(causal,key_chunk):
    q,k,v,_,_=tensors(b=2)
    out,lse=attention_with_lse(q,k,v,causal=causal,key_chunk=key_chunk,query_chunk=2)
    ref,rl=dense_attn(q,k,v,causal)
    torch.testing.assert_close(out,ref,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(lse,rl,atol=1e-12,rtol=1e-12)


def test_joint_attention_affine_exact():
    q,k,v,kt,vt=tensors()
    h,d=k.shape[2:]
    reader=Reader(torch.randn(h,d,d,dtype=q.dtype),torch.randn(h,d,d,dtype=q.dtype),
                  torch.randn(h,d,dtype=q.dtype),torch.randn(h,d,dtype=q.dtype))
    kh=torch.einsum("bthi,hij->bthj",k,reader.A)+reader.bk
    vh=torch.einsum("bthi,hij->bthj",v,reader.B)+reader.bv
    actual=segmented_attention(q,k,v,kt,vt,reader=reader)
    # Concatenated prefix and bottom-right-causal private tail reference.
    expected,_=dense_attn(q,torch.cat((kh,kt),1),torch.cat((vh,vt),1),True)
    torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-12)


def test_omitting_key_bias_changes_mass():
    q,k,v,kt,vt=tensors()
    h,d=k.shape[2:]
    i=torch.eye(d,dtype=q.dtype).repeat(h,1,1)
    r=Reader(i,i,torch.ones(h,d,dtype=q.dtype)*3,torch.zeros(h,d,dtype=q.dtype))
    out=segmented_attention(q,k,v,kt,vt,reader=r)
    wrong=segmented_attention(q,k,v,kt,vt)
    assert (out-wrong).abs().max()>0.1


def test_value_bias_scaled_by_prefix_mass():
    q,k,v,kt,vt=tensors()
    h,d=k.shape[2:]
    i=torch.eye(d,dtype=q.dtype).repeat(h,1,1)
    r=Reader(i,i,torch.zeros(h,d,dtype=q.dtype),torch.ones(h,d,dtype=q.dtype))
    actual=segmented_attention(q,k,v,kt,vt,reader=r)
    wrong=segmented_attention(q,k,v,kt,vt)+1
    assert (actual-wrong).abs().max()>0.05


def test_identity_segmentation():
    q,k,v,kt,vt=tensors()
    out=segmented_attention(q,k,v,kt,vt)
    ref,_=dense_attn(q,torch.cat((k,kt),1),torch.cat((v,vt),1),True)
    torch.testing.assert_close(out,ref,atol=1e-12,rtol=1e-12)


def test_gqa_group_specific_readers():
    q,k,v,kt,vt=tensors()
    h,d=k.shape[2:]
    aa=torch.eye(d,dtype=q.dtype).repeat(h,1,1)
    aa[1]*=2
    r=Reader(aa,aa,torch.zeros(h,d,dtype=q.dtype),torch.zeros(h,d,dtype=q.dtype))
    transformed=query_transform(q,r)
    torch.testing.assert_close(transformed[:,:,:3],q[:,:,:3])
    torch.testing.assert_close(transformed[:,:,3:],2*q[:,:,3:])


def test_rope_commutation_split_half():
    torch.manual_seed(1)
    x=torch.randn(128,8,dtype=torch.float64)
    y=torch.randn_like(x)
    a=commuting_rope_fit(x,y)
    theta=torch.randn(128,4,dtype=x.dtype)
    c=torch.cat((theta.cos(),theta.cos()),-1)
    s=torch.cat((theta.sin(),theta.sin()),-1)
    rotate=lambda z:z*c+rotate_half(z)*s
    torch.testing.assert_close(rotate(x@a),rotate(x)@a,rtol=1e-12,atol=1e-12)


def test_dense_affine_fit_recovery():
    torch.manual_seed(2)
    x=torch.randn(512,8)
    a=torch.randn(8,8);b=torch.randn(8)
    y=x@a+b
    af,bf,_=fit_key_reader(x,y,torch.randn(128,8),mode="dense",ridge=1e-8)
    torch.testing.assert_close(af,a,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(bf,b,rtol=1e-5,atol=1e-5)


def test_rank_constraint_and_improvement():
    torch.manual_seed(3)
    x=torch.randn(500,8,dtype=torch.float64)
    y=torch.randn(500,8,dtype=torch.float64)
    metric=torch.diag(torch.arange(1,9,dtype=torch.float64))
    losses=[]
    for r in (0,1,3,8):
        delta=weighted_reduced_rank(x,y,metric,r,1e-3)
        assert int(torch.linalg.matrix_rank(delta,atol=1e-8))<=r
        residual=x@delta-y
        losses.append(float((residual@metric*residual).sum()))
    assert all(a>=b-1e-8 for a,b in zip(losses,losses[1:]))


def test_value_attention_regression():
    torch.manual_seed(4)
    us=torch.randn(512,8);uv=torch.randn_like(us)
    b=torch.randn(8,8);bias=torch.randn(8)
    ut=us@b+bias
    mass=torch.rand(512)*0.9+0.05
    bf,cf=fit_value_from_attention(us,ut,uv,mass,mass,ridge=1e-8)
    torch.testing.assert_close(bf,b,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(cf,bias,atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize("sources,period",[([1,1,2,3],2),([0,2,2,3],2),([0,2,1,3],4),([0,1,9,3],4)])
def test_bad_source_maps(sources,period):
    with pytest.raises(ValueError):validate_sources(sources,period=period)


def test_valid_future_source():
    validate_sources([0,1,3,3,4,5,6,7],period=8)


class FakeLoader:
    def __init__(self,k,v,layers=4):
        self.k,self.v=k,v;self._loaded={};self.calls=[];self.schedules=[]
        self._selected_by_layer=[list(range(k.shape[0])) for _ in range(layers)]
    def resolve(self,lid):
        self.calls.append(lid);return self.k,self.v
    def schedule(self,lid):self.schedules.append(lid)
    def prefetch_selector_keys(self,lid):pass


def test_physical_resolve_once_and_alias():
    _,k,v,_,_=tensors()
    loader=FakeLoader(k[0],v[0])
    state=RequestState([0,1,2,2],{},"reference")
    state.schedule_window(loader,0,4)
    a=state.resolve(loader,2,dtype=k.dtype,device=k.device)
    b=state.resolve(loader,3,dtype=k.dtype,device=k.device)
    assert a[0].data_ptr()==b[0].data_ptr()
    assert a[1].data_ptr()==b[1].data_ptr()
    assert loader.calls==[2] and loader.schedules==[0,1,2]


def test_mismatched_selection_rejected():
    _,k,v,_,_=tensors()
    loader=FakeLoader(k[0],v[0]);loader._selected_by_layer[3]=[0]
    with pytest.raises(RuntimeError):
        RequestState([0,1,2,2],{},"reference").schedule_window(loader,0,4)


class TinyCache:
    def __init__(self,n):self.data=[None]*n
    def get_seq_length(self,lid=0):return 0 if self.data[lid] is None else self.data[lid][0].shape[2]
    def update(self,k,v,lid,extra):
        old=self.data[lid]
        if old is not None:k,v=torch.cat((old[0],k),2),torch.cat((old[1],v),2)
        self.data[lid]=(k,v);return k,v
    def fork(self):
        c=TinyCache(len(self.data));c.data=[(k.clone(),v.clone()) for k,v in self.data];return c


def test_decoder_private_tail_fork_and_shared_lifetime(monkeypatch):
    torch.manual_seed(5)
    hq,hk,d,nlayers=4,2,4,4
    hidden=hq*d
    def linear(out):return torch.nn.Linear(hidden,out,dtype=torch.float64)
    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embed_tokens=torch.nn.Embedding(23,hidden,dtype=torch.float64)
            self.norm=torch.nn.Identity();self.layers=[]
            for i in range(nlayers):
                attn=types.SimpleNamespace(q_proj=linear(hq*d),k_proj=linear(hk*d),v_proj=linear(hk*d),o_proj=linear(hidden))
                self.layers.append(types.SimpleNamespace(input_layernorm=torch.nn.Identity(),
                    post_attention_layernorm=torch.nn.Identity(),self_attn=attn,mlp=linear(hidden)))
        def rotary_emb(self,x,positions):
            theta=positions.to(x.dtype)[...,None]*torch.tensor([0.1,0.5],dtype=x.dtype)
            return torch.cat((theta.cos(),theta.cos()),-1),torch.cat((theta.sin(),theta.sin()),-1)
    model=types.SimpleNamespace(model=Core(),lm_head=linear(23))
    k=torch.randn(9,hk,d,dtype=torch.float64);v=torch.randn_like(k)
    loader=FakeLoader(k,v)
    runner=types.ModuleType("contiguous_fuxian.flexgen_qwen_reprefill")
    runner.configure_online_layer_selection=lambda **kw:4
    monkeypatch.setitem(sys.modules,"contiguous_fuxian.flexgen_qwen_reprefill",runner)
    state=RequestState([0,1,2,2],{},"reference",geometry={"layers":4,"q_heads":hq,"kv_heads":hk,"head_dim":d})
    token=_ACTIVE.set(state)
    cache=TinyCache(4)
    base=dict(model=model,loader=loader,period_size=4,subperiod_size=2,impress_selection_period_size=4)
    try:
        result=decoder_logits(input_ids=torch.tensor([[1,2,3]]),cache=cache,position_start=9,**base)
        assert result.shape==(1,3,23)
        assert cache.get_seq_length()==3 # private only, NOT 12.
        fork=cache.fork()
        out1=decoder_logits(input_ids=torch.tensor([[4]]),cache=cache,position_start=12,**base)
        out2=decoder_logits(input_ids=torch.tensor([[4]]),cache=fork,position_start=12,**base)
        torch.testing.assert_close(out1,out2)
        assert loader.calls==[0,1,2] # neither decode nor label fork rereads prefix.
        assert cache.data[0][0].data_ptr()!=fork.data[0][0].data_ptr()
    finally:_ACTIVE.reset(token)


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("flash_attn") is None,
                    reason="CUDA and FlashAttention-2 unavailable")
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
def test_flash2_segmented_matches_reference(dtype):
    q,k,v,kt,vt=tensors(dtype=dtype,device="cuda",d=128,t=5,n=67,tail=9)
    h,d=k.shape[2:]
    eye=torch.eye(d,device="cuda",dtype=dtype).repeat(h,1,1)
    reader=Reader(eye*0.8,eye*1.1,torch.randn(h,d,device="cuda",dtype=dtype)*0.1,
                  torch.randn(h,d,device="cuda",dtype=dtype)*0.1)
    actual=segmented_attention(q,k,v,kt,vt,reader=reader,backend="flash2")
    expected=segmented_attention(q,k,v,kt,vt,reader=reader,backend="reference")
    torch.testing.assert_close(actual,expected,atol=0.025 if dtype==torch.bfloat16 else 0.004,rtol=0.03)


def test_compiler_end_to_end_synthetic(tmp_path):
    import json
    from contiguous_fuxian.sharedkv.compile import main
    from contiguous_fuxian.sharedkv.io import TRACE_FORMAT,BASE_COMMIT,load_safe
    torch.manual_seed(8)
    task="demo";n,h,d,hq,layers=32,2,4,4,4
    store=tmp_path/"store"/task;store.mkdir(parents=True)
    meta={"dtype":"bfloat16","layout":"token,kv_head,head_dim","task":task,
          "prefix_tokens":n,"kv_heads":h,"head_dim":d,"layers":layers,"token_hash":"synthetic"}
    (store/"metadata.json").write_text(json.dumps(meta))
    k=torch.randn(n,h,d);v=torch.randn(n,h,d)
    # Equal source/target ensures the pipeline can pass, without claiming real
    # Qwen layers have this property. Roundtrip is exactly the documented store.
    for lid in range(layers):
        for kind,x in (("key",k),("value",v)):
            (store/f"layer_{lid:02d}_{kind}.bf16").write_bytes(x.bfloat16().view(torch.uint8).numpy().tobytes())
    common={"format":TRACE_FORMAT,"base_commit":BASE_COMMIT,"task":task,"token_hash":"synthetic",
            "prefix_tokens":n,"geometry":{"layers":layers,"q_heads":hq,"kv_heads":h,"head_dim":d},
            "model_identity":{"synthetic":True},"period":4,"keep_ratio":0.5}
    for split in ("fit","val"):
        path=tmp_path/split/task;path.mkdir(parents=True)
        for idx in range(3):
            q=torch.randn(8,hq,d)
            layer_rows={lid:{"q":q,"tail_output":torch.randn_like(q),"tail_lse":torch.ones(8,hq),
                              "selected_tokens":torch.arange(16)} for lid in range(layers)}
            # Tail differs by layer but teacher and student for each target use
            # the SAME private segment, as the compiler contract specifies.
            torch.save({**common,"query_hash":f"{split}-{idx}","layers":layer_rows},path/f"{idx}.pt")
    out=tmp_path/"adapters"/"demo.pt"
    main(["--store-root",str(tmp_path/"store"),"--task",task,"--fit-traces",str(tmp_path/"fit"),
          "--validation-traces",str(tmp_path/"val"),"--pairs","2:3","--key-mode","dense",
          "--ridge","1e-6","--output",str(out)])
    bundle=load_safe(out)
    assert bundle["sources"]==[0,1,2,2]
    assert bundle["accepted_consumers"]==1
    assert bundle["reports"]["3"]["output_relative_rmse"]<1e-4
    assert bundle["end_to_end_validated"] is False
    assert len(bundle["calibration_query_hashes"])==6
    # Whole-request overlap must fail closed.
    with pytest.raises(ValueError,match="overlap"):
        main(["--store-root",str(tmp_path/"store"),"--task",task,"--fit-traces",str(tmp_path/"fit"),
          "--validation-traces",str(tmp_path/"fit"),"--pairs","2:3","--output",str(tmp_path/"other.pt")])
