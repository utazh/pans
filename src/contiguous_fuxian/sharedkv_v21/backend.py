"""Request-local, bounded reuse of vLLM FA2 cu_seqlens. No KV head expansion.

This is an allocation/launch-count optimization, not an acceleration claim.
All attention calls remain inside the timed decoder. Cache misses are charged.
"""
from collections import OrderedDict
from contextvars import ContextVar
import torch
from contiguous_fuxian.sharedkv.core import attention_with_lse as reference_or_v1

_ACTIVE_POOL = ContextVar('sharedkv_v21_metadata', default=None)

class LengthPool:
    def __init__(self, max_entries=64):
        if max_entries < 1: raise ValueError('max_entries must be positive')
        self.max_entries=max_entries
        self.data=OrderedDict(); self.hits=0; self.misses=0
    def get(self, batch, length, device):
        if batch < 1 or length < 1: raise ValueError('nonempty sequences required')
        device=torch.device(device)
        key=(batch,length,str(device))
        if key in self.data:
            self.hits+=1; self.data.move_to_end(key); return self.data[key]
        self.misses+=1
        if len(self.data)>=self.max_entries: self.data.popitem(last=False)
        x=torch.arange(0,(batch+1)*length,length,dtype=torch.int32,device=device)
        self.data[key]=x; return x
    def stats(self):
        return {'cu_seqlens_cache_hits':self.hits,'cu_seqlens_cache_misses':self.misses,
                'cu_seqlens_resident_bytes':sum(x.numel()*x.element_size() for x in self.data.values())}


def attention(q,k,v,*,causal=False,backend='reference',q_positions=None,**kwargs):
    if backend!='vllm2':
        return reference_or_v1(q,k,v,causal=causal,backend=backend,q_positions=q_positions,**kwargs)
    from contiguous_fuxian.sharedkv.core import _geometry
    _geometry(q,k,v)
    if q_positions is not None or q.device.type!='cuda' or q.dtype not in (torch.float16,torch.bfloat16):
        raise ValueError('vllm2 requires CUDA FP16/BF16, contiguous query positions')
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    b,t,hq,d=q.shape; n,hk=k.shape[1:3]
    pool=_ACTIVE_POOL.get()
    if pool is None:
        return reference_or_v1(q,k,v,causal=causal,backend='vllm2')
    cuq=pool.get(b,t,q.device); cuk=pool.get(b,n,q.device)
    result=flash_attn_varlen_func(q.reshape(b*t,hq,d),k.reshape(b*n,hk,d),v.reshape(b*n,hk,d),
        max_seqlen_q=t,cu_seqlens_q=cuq,max_seqlen_k=n,cu_seqlens_k=cuk,
        causal=causal,softmax_scale=d**-0.5,return_softmax_lse=True,fa_version=2)
    if not isinstance(result,(tuple,list)) or len(result)!=2:
        raise RuntimeError('Audited vLLM FA2 return API changed')
    out,lse=result
    if out.shape!=(b*t,hq,d) or lse.numel()!=b*hq*t:
        raise RuntimeError('Unexpected vLLM FA2 output/LSE shape')
    if b!=1: raise ValueError('Only batch=1 has been audited; no implicit batch LSE layout assumption')
    return out.reshape_as(q),lse.reshape(1,hq,t)
