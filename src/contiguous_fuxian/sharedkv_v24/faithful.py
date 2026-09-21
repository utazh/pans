
"""Server-calibration replay matching the unchanged v21 hybrid/packed scorer.

FA2 forward is reused exactly. Backward differentiates the mathematical
softmax attention (including LSE gradients) in FP32, as an explicit VJP.
Private KV is recomputed and threaded functionally through incremental decode.
This module does not patch the online inference kernel or cold protocol.
"""
import torch
from torch.utils.checkpoint import checkpoint
from contiguous_fuxian.sharedkv.core import apply_rope
from contiguous_fuxian.sharedkv_v21.ops import prepare_query,finish

def probabilities(q,k,causal):
 b,t,hq,d=q.shape;n,hk=k.shape[1:3];g=hq//hk
 dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
 qr=q.to(dtype).reshape(b,t,hk,g,d).permute(0,2,3,1,4)
 kr=k.to(dtype).permute(0,2,1,3).unsqueeze(2)
 scores=qr@kr.transpose(-1,-2)*(d**-.5)
 if causal:
  visible=torch.arange(n,device=q.device)[None,:]<=n-t+torch.arange(t,device=q.device)[:,None]
  scores=scores.masked_fill(~visible,-torch.inf)
 return qr,kr,scores.softmax(-1),scores.logsumexp(-1).reshape(b,hq,t)

class AttentionVJP(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,causal):
  if q.device.type=='cuda':
   from contiguous_fuxian.sharedkv_v21.backend import attention
   out,lse=attention(q,k,v,causal=causal,backend='vllm2')
  else:
   _,_,p,lse=probabilities(q,k,causal);b,t,hq,d=q.shape
   u=p@v.to(p.dtype).permute(0,2,1,3).unsqueeze(2)
   out=u.permute(0,3,1,2,4).reshape_as(q).to(q.dtype)
  ctx.save_for_backward(q,k,v);ctx.causal=causal
  return out,lse
 @staticmethod
 def backward(ctx,du,dz):
  q,k,v=ctx.saved_tensors;b,t,hq,d=q.shape;n,hk=k.shape[1:3];g=hq//hk
  qr,kr,p,_=probabilities(q,k,ctx.causal)
  go=torch.zeros_like(q) if du is None else du
  go=go.to(p.dtype).reshape(b,t,hk,g,d).permute(0,2,3,1,4)
  vr=v.to(p.dtype).permute(0,2,1,3).unsqueeze(2)
  dp=go@vr.transpose(-1,-2)
  ds=p*(dp-(dp*p).sum(-1,keepdim=True))
  if dz is not None:ds=ds+p*dz.to(p.dtype).reshape(b,hk,g,t,1)
  dq=(ds@kr)*(d**-.5)
  dk=(ds.transpose(-1,-2)@qr).sum(2)*(d**-.5)
  dv=(p.transpose(-1,-2)@go).sum(2)
  return (dq.permute(0,3,1,2,4).reshape_as(q).to(q.dtype),
          dk.permute(0,2,1,3).to(k.dtype),dv.permute(0,2,1,3).to(v.dtype),None)

def attention(q,k,v,causal=False):
 return AttentionVJP.apply(q,k,v,causal)

def decoder(model,ids,position,cache,sources,bank,readers,checkpoint_layers=True,layout='hybrid'):
 hidden=model.model.embed_tokens(ids);b,t,_=hidden.shape
 pos=torch.arange(position,position+t,device=hidden.device)[None]
 cos,sin=model.model.rotary_emb(hidden,pos)
 first=min((int(i) for i in readers),default=len(sources));next_cache=[]
 for lid,layer in enumerate(model.model.layers):
  pk,pv=bank[sources[lid]];prior=cache[lid] if cache else None
  reader=readers[str(lid)] if str(lid) in readers else None
  def step(x,layer=layer,pk=pk,pv=pv,prior=prior,reader=reader):
   attn=layer.self_attn;cfg=attn.config
   hq,hk,d=cfg.num_attention_heads,cfg.num_key_value_heads,attn.head_dim
   z=layer.input_layernorm(x)
   q=apply_rope(attn.q_proj(z).view(b,t,hq,d),cos,sin)
   k=apply_rope(attn.k_proj(z).view(b,t,hk,d),cos,sin)
   v=attn.v_proj(z).view(b,t,hk,d)
   tk,tv=(k,v) if prior is None else (torch.cat((prior[0],k),1),torch.cat((prior[1],v),1))
   if reader is None and layout=='hybrid':
    out,_=attention(q,torch.cat((pk,tk),1),torch.cat((pv,tv),1),True)
   else:
    if reader is None:
     qp=q;offset=0
    else:
     p=reader.effective()
     qp,offset=prepare_query(q,p.A.to(q.dtype).transpose(-1,-2).contiguous(),p.bk.float())
    up,lp=attention(qp,pk,pv,False);ut,lt=attention(q,tk,tv,True)
    if reader is None:
     beta=(lp-torch.logaddexp(lp,lt)).exp().transpose(1,2).unsqueeze(-1)
     out=(beta*up.float()+(1-beta)*ut.float()).to(q.dtype)
    else:out=finish(up,lp,ut,lt,offset,p.B.to(q.dtype).contiguous(),p.bv.to(q.dtype))
   mid=x+attn.o_proj(out.reshape(b,t,-1).contiguous())
   return mid+layer.mlp(layer.post_attention_layernorm(mid)),tk,tv
  if checkpoint_layers and torch.is_grad_enabled() and lid>=first:
   hidden,tk,tv=checkpoint(step,hidden,use_reentrant=False)
  else:hidden,tk,tv=step(hidden)
  next_cache.append((tk,tv))
 return model.lm_head(model.model.norm(hidden)),next_cache

def continuation(model,row,label,sources,bank,readers,*,checkpoint_layers=True,max_sequence=256):
 query=row['query_token_ids'];candidate=row['label_ids'][label]
 if len(query)+len(candidate)-1>max_sequence:raise ValueError('Sequence exceeds explicit memory limit; no truncation')
 if row['teacher_backend']!='vllm2' or row['teacher_layout']!='hybrid':
  raise ValueError('Server faithful replay audited only for vllm2 hybrid')
 device=next(model.parameters()).device
 logits,cache=decoder(model,torch.tensor([query],device=device),row['prefix_tokens'],None,sources,bank,readers,checkpoint_layers)
 scores=[logits[0,-1].float().log_softmax(-1)[candidate[0]]]
 for i,(token,target) in enumerate(zip(candidate,candidate[1:])):
  logits,cache=decoder(model,torch.tensor([[token]],device=device),row['prefix_tokens']+len(query)+i,cache,sources,bank,readers,checkpoint_layers)
  scores.append(logits[0,-1].float().log_softmax(-1)[target])
 return torch.stack(scores)
