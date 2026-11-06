import copy
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache
from contiguous_fuxian.sharedkv.core import attention_with_lse
from contiguous_fuxian.sharedkv.runtime import RequestState, _ACTIVE, decoder_logits

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
@pytest.mark.parametrize("batch,t,n,causal",[(1,1,71,True),(1,7,71,True),(2,5,71,False),(2,5,71,True)])
def test_vllm_qwen_gqa_lse(batch,t,n,causal):
    torch.manual_seed(91)
    q=torch.randn(batch,t,28,128,device="cuda",dtype=torch.bfloat16)
    k=torch.randn(batch,n,4,128,device="cuda",dtype=q.dtype)
    v=torch.randn_like(k)
    a,l=attention_with_lse(q,k,v,backend="vllm2",causal=causal)
    b,m=attention_with_lse(q,k,v,backend="reference",causal=causal)
    torch.testing.assert_close(a,b,atol=.016,rtol=.03)
    torch.testing.assert_close(l,m,atol=2e-5,rtol=2e-5)

def test_actual_qwen_identity_prefill_decode_and_private_cache(monkeypatch):
    torch.manual_seed(912)
    cfg=Qwen2Config(vocab_size=128,hidden_size=64,intermediate_size=128,
                   num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,
                   max_position_embeddings=256)
    cfg._attn_implementation="eager"
    model=Qwen2ForCausalLM(cfg).eval()
    prefix=torch.tensor([[2,3,4,5,6,7]])
    with torch.inference_mode():
        teacher=model(prefix,use_cache=True).past_key_values
        pairs=[(x.keys.transpose(1,2)[0].clone(),x.values.transpose(1,2)[0].clone()) for x in teacher.layers]
        class Loader:
            _selected_by_layer=[list(range(6)) for _ in range(3)]
            _loaded={}
            def schedule(self,l): pass
            def prefetch_selector_keys(self,l): pass
            def resolve(self,l): return pairs[l]
        loader=Loader()
        monkeypatch.setattr("contiguous_fuxian.flexgen_qwen_reprefill.configure_online_layer_selection",lambda **kw:1)
        g={"layers":3,"q_heads":4,"kv_heads":2,"head_dim":16}
        state=RequestState(list(range(3)),{},"reference",geometry=g)
        token=_ACTIVE.set(state)
        cache=DynamicCache(config=cfg)
        try:
            for start,ids in [(6,[[11,12,13]]),(9,[[14]]),(10,[[15]])]:
                x=torch.tensor(ids)
                expected=model(x,past_key_values=teacher,use_cache=True).logits
                actual=decoder_logits(model=model,input_ids=x,cache=cache,position_start=start,
                                      loader=loader,period_size=1,subperiod_size=1)
                torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
                assert cache.get_seq_length()==start+x.shape[1]-6
            assert state.resolve_calls==3
        finally: _ACTIVE.reset(token)


def test_diagnostic_bundle_requires_explicit_opt_in(monkeypatch):
    from types import SimpleNamespace
    from contiguous_fuxian.sharedkv.runtime import SharedRuntime
    config=Qwen2Config(vocab_size=128,hidden_size=64,intermediate_size=128,
                      num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    model=Qwen2ForCausalLM(config).to(torch.bfloat16)
    loader=SimpleNamespace(_info=SimpleNamespace(task="demo"))
    monkeypatch.setattr("contiguous_fuxian.sharedkv.runtime.load_safe",
                        lambda path: {"diagnostic_only":True})
    runtime=SharedRuntime(mode="shared",backend="reference",adapter_dir="unused")
    with pytest.raises(ValueError,match="Locally rejected"):
        runtime._prepare(model,loader,8)

@pytest.mark.parametrize("fail",[False,True])
def test_fixed_selection_restores_class_method_without_reference_cycle(fail):
    import weakref
    from contiguous_fuxian.sharedkv.selection import fixed_selection
    class Loader:
        def configure_impress_layer(self, **kw): return kw
    loader=Loader()
    weak=weakref.ref(loader)
    try:
        with fixed_selection(loader,[[2,4]]):
            assert loader.configure_impress_layer(layer=0,selected_tokens=[1])["selected_tokens"]==[2,4]
            if fail:raise RuntimeError("test")
    except RuntimeError: pass
    assert "configure_impress_layer" not in vars(loader)
    del loader
    assert weak() is None

def test_fixed_selection_preserves_preexisting_instance_override():
    from contiguous_fuxian.sharedkv.selection import fixed_selection
    class Loader: pass
    loader=Loader()
    original=lambda **kw:kw
    loader.configure_impress_layer=original
    with fixed_selection(loader,[[2,4]]):
        assert loader.configure_impress_layer(layer=0,selected_tokens=[1])["selected_tokens"]==[2,4]
    assert loader.configure_impress_layer is original
