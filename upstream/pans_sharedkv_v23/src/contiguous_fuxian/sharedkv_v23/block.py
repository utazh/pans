"""Differentiable FULL query + forced label-prefix replay through frozen layers.

No consumer prefix KV is materialized. Only readers are trainable. This is a
calibration backend (FP32 softmax/BF16 boundaries), not a latency backend.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Parameters:
    A: torch.Tensor
    B: torch.Tensor
    bk: torch.Tensor
    bv: torch.Tensor

class TrainableReader(nn.Module):
    def __init__(self, reader, ste=True):
        super().__init__();self.ste=ste
        for n in ('A','B','bk','bv'):
            self.register_parameter(n,nn.Parameter(getattr(reader,n).detach().clone().float()))
    def effective(self):
        values=[]
        for n in ('A','B','bk','bv'):
            v=getattr(self,n)
            if self.ste:v=v+(v.to(torch.bfloat16).float()-v).detach()
            values.append(v)
        return Parameters(*values)
    def export(self):
        from contiguous_fuxian.sharedkv.core import Reader
        return Reader(*(getattr(self,n).detach().to(torch.bfloat16).float().cpu()
                        for n in ('A','B','bk','bv')))


def rope(x, cos, sin):
    a,b=x.chunk(2,-1)
    return x*cos.unsqueeze(2)+torch.cat((-b,a),-1)*sin.unsqueeze(2)


def segment(q,k,v,causal=False):
    b,t,hq,d=q.shape;n,hk=k.shape[1:3]
    if n<1 or hq%hk or k.shape!=v.shape:raise ValueError('Invalid attention geometry')
    dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
    qr=q.to(dtype).reshape(b,t,hk,hq//hk,d).permute(0,2,3,1,4)
    scores=torch.matmul(qr,k.to(dtype).permute(0,2,3,1).unsqueeze(2))*(d**-.5)
    if causal:
        visible=torch.arange(n,device=q.device)[None,:] <= (n-t+torch.arange(t,device=q.device))[:,None]
        scores=scores.masked_fill(~visible,-torch.inf)
    p=scores.softmax(-1);z=scores.logsumexp(-1).reshape(b,hq,t)
    u=torch.matmul(p,v.to(dtype).permute(0,2,1,3).unsqueeze(2))
    u=u.permute(0,3,1,2,4).reshape(b,t,hq,d).to(q.dtype)
    return u,z


def shared_attention(q,pk,pv,tk,tv,reader=None):
    b,t,hq,d=q.shape;hk=pk.shape[2];g=hq//hk
    if reader is not None:
        # Preserve the published BF16 rounding points (matmul then bias).
        p=Parameters(*(getattr(reader,n).to(device=q.device,dtype=q.dtype) for n in ('A','B','bk','bv')))
        qp=torch.einsum('bthgj,hij->bthgi',q.reshape(b,t,hk,g,d),p.A).reshape_as(q)
    else:qp=q
    up,lp=segment(qp,pk,pv)
    if reader is not None:
        up=torch.einsum('bthgi,hij->bthgj',up.reshape(b,t,hk,g,d),p.B)
        up=(up+p.bv[None,None,:,None,:]).reshape_as(q)
        dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
        off=torch.einsum('bthgd,hd->bthg',q.reshape(b,t,hk,g,d).to(dtype),reader.bk.to(dtype))
        lp=lp+off.reshape(b,t,hq).transpose(1,2)*(d**-.5)
    ut,lt=segment(q,tk,tv,True)
    beta=(lp-torch.logaddexp(lp,lt)).exp()
    dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
    w=beta.transpose(1,2).unsqueeze(-1).to(dtype)
    return (w*up.to(dtype)+(1-w)*ut.to(dtype)).to(q.dtype),beta


def layer_forward(layer,x,pk,pv,cos,sin,reader=None,return_details=False):
    attn=layer.self_attn;cfg=attn.config
    hq,hk=int(cfg.num_attention_heads),int(cfg.num_key_value_heads)
    d=int(getattr(attn,'head_dim',cfg.hidden_size//hq));b,t,_=x.shape
    z=layer.input_layernorm(x)
    q=rope(attn.q_proj(z).reshape(b,t,hq,d),cos,sin)
    k=rope(attn.k_proj(z).reshape(b,t,hk,d),cos,sin)
    v=attn.v_proj(z).reshape(b,t,hk,d)
    o,mass=shared_attention(q,pk,pv,k,v,reader)
    mid=x+attn.o_proj(o.reshape(b,t,-1))
    y=mid+layer.mlp(layer.post_attention_layernorm(mid))
    return (y,o,mass) if return_details else y


def replay_group(model,group,hidden,positions,prefix,readers,*,native=False,details=False):
    """prefix[lid]=(selected native K,V). Source tensors reused by reference."""
    cos,sin=model.model.rotary_emb(hidden,positions)
    x=hidden;outs={};infos={}
    for lid in range(group.start,group.end):
        r=None if native or lid not in group.consumers else readers[str(lid)].effective()
        pk,pv=prefix[lid if native or r is None else group.source]
        x,o,m=layer_forward(model.model.layers[lid],x,pk,pv,cos,sin,r,True)
        outs[lid]=x
        if details:infos[lid]=(o,m)
    return x,outs,infos


def boundary_loss(student,teacher,teacher_input,query_tokens,*,last_weight=0.5,suffix_weight=1.0):
    """Normalize to the teacher GROUP UPDATE, not its large residual stream."""
    s,t,h=student.float(),teacher.float(),teacher_input.float()
    scale=(t-h).square().mean().clamp_min(1e-8)
    base=(s[:,:query_tokens]-t[:,:query_tokens]).square().mean()/scale
    last=(s[:,query_tokens-1]-t[:,query_tokens-1]).square().mean()/scale
    suffix=(s[:,query_tokens:]-t[:,query_tokens:]).square().mean()/scale if s.shape[1]>query_tokens else base*0
    return base+last_weight*last+suffix_weight*suffix, {'base':base,'last':last,'suffix':suffix}
