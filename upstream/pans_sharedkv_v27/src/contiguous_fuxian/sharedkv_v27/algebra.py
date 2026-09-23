"""Dense mathematical reference. NEVER a timing/backend implementation."""
from __future__ import annotations
import torch

def _repeat(k,hq):
 if k.ndim!=4 or hq%k.shape[2]:raise ValueError('Bad GQA')
 return k.repeat_interleave(hq//k.shape[2],dim=2)
def attn(q,k,v,causal=False):
 if k.shape!=v.shape or q.shape[-1]!=k.shape[-1] or q.shape[0]!=k.shape[0]:raise ValueError('Bad shapes')
 kk,vv=_repeat(k,q.shape[2]),_repeat(v,q.shape[2])
 scores=torch.einsum('bthd,bnhd->bhtn',q,kk)*q.shape[-1]**-.5
 if causal:
  n,t=k.shape[1],q.shape[1]
  if n<t:raise ValueError('Private tail must include all current queries')
  visible=torch.arange(n,device=q.device)[None,:]<=n-t+torch.arange(t,device=q.device)[:,None]
  scores=scores.masked_fill(~visible,-torch.inf)
 p=scores.softmax(-1)
 return torch.einsum('bhtn,bnhd->bthd',p,vv),scores.logsumexp(-1)
def affine_kv(k,v,A,B,bk,bv):
 return (torch.einsum('bnhd,hde->bnhe',k,A)+bk[None,None],
         torch.einsum('bnhd,hde->bnhe',v,B)+bv[None,None])
def read(q,k,v,A,B,bk,bv):
 b,t,hq,d=q.shape;hk=A.shape[0];x=q.reshape(b,t,hk,hq//hk,d)
 qr=torch.einsum('bthgj,hij->bthgi',x,A).reshape_as(q)
 out,z=attn(qr,k,v)
 out=(torch.einsum('bthgi,hij->bthgj',out.reshape_as(x),B)+bv[None,None,:,None]).reshape_as(q)
 off=torch.einsum('bthgd,hd->bthg',x,bk).reshape(b,t,hq).transpose(1,2)*d**-.5
 return out,z+off
def merge(up,zp,ut,zt):
 beta=(zp-torch.logaddexp(zp,zt)).exp().transpose(1,2).unsqueeze(-1)
 return beta*up+(1-beta)*ut

def frozen_r0(q,k,v,tk,tv,A,B,bk,bv):
 up,zp=read(q,k,v,A,B,bk,bv);ut,zt=attn(q,tk,tv,True)
 return merge(up,zp,ut,zt)
def proxy_split(q,k,v,tk,tv,A,B,bk,bv,cut):
 if not 0<cut<k.shape[1]:raise ValueError('Both segments must be nonempty')
 up,zp=read(q,k[:,:cut],v[:,:cut],A,B,bk,bv)
 rk,rv=affine_kv(k[:,cut:],v[:,cut:],A,B,bk,bv)
 ut,zt=attn(q,torch.cat((rk,tk),1),torch.cat((rv,tv),1),True)
 return merge(up,zp,ut,zt)
