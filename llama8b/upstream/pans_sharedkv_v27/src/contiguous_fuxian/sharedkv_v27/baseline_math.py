"""Clean-room PRIMARY-PAPER references, not full external serving systems.

CLF follows xKV shared token basis Xcat ~= U S V^T, pre-RoPE K and unrotated V.
The KVSharer-style search primitive follows descending dissimilarity + final
hidden cosine acceptance. Its evaluator and the prefix-specific calibration
semantics must be supplied; it is NOT the official pretrained-model notebook.
"""
from __future__ import annotations
import math
import torch
from .common import validate_sources

def rotary(x,positions,theta=1e6,inverse=False):
 if x.ndim!=3 or x.shape[-1]%2 or positions.shape!=(x.shape[0],) or theta<=0:raise ValueError('Full fixed split-half RoPE only')
 d=x.shape[-1];work=x.double() if x.dtype==torch.float64 else x.float()
 freq=1/(theta**(torch.arange(0,d,2,device=x.device,dtype=work.dtype)/d))
 angles=positions.to(work.dtype)[:,None]*freq[None,:]
 cs=torch.cat((angles.cos(),angles.cos()),-1)[:,None,:]
 sn=torch.cat((angles.sin(),angles.sin()),-1)[:,None,:]
 if inverse:sn=-sn
 a,b=work.chunk(2,-1)
 return (work*cs+torch.cat((-b,a),-1)*sn).to(x.dtype)

def factorize(tensors,rank,*,seed=27,oversample=8,exact=False):
 if not tensors or any(t.ndim!=3 or t.shape!=tensors[0].shape for t in tensors):raise ValueError('Equal [N,Hkv,D] tensors required')
 n,h,d=tensors[0].shape;dtype=torch.float64 if tensors[0].dtype==torch.float64 else torch.float32
 x=torch.cat([t.to(dtype).reshape(n,h*d) for t in tensors],1)
 if not 1<=rank<=min(x.shape):raise ValueError('Invalid CLF rank')
 if exact:
  u,s,vh=torch.linalg.svd(x,full_matrices=False);u,s,v=u[:,:rank],s[:rank],vh[:rank].T
 else:
  devices=[x.device.index] if x.device.type=='cuda' else []
  with torch.random.fork_rng(devices=devices):
   torch.manual_seed(seed);u,s,v=torch.svd_lowrank(x,q=min(min(x.shape),rank+oversample),niter=4)
  u,s,v=u[:,:rank],s[:rank],v[:,:rank]
 left=(u*s[None,:]).contiguous()
 rights=[v[i*h*d:(i+1)*h*d].T.contiguous() for i in range(len(tensors))]
 return {'left':left,'rights':rights,'geometry':[n,h,d],'rank':rank}

def reconstruct(factor,layer,ids):
 n,h,d=factor['geometry']
 if not 0<=layer<len(factor['rights']) or ids.ndim!=1 or ids.dtype!=torch.long:raise ValueError('Bad indices')
 if ids.numel()==0 or int(ids.min())<0 or int(ids.max())>=n:raise ValueError('Empty/out-of-range support')
 return (factor['left'].index_select(0,ids)@factor['rights'][layer]).reshape(len(ids),h,d)

def factor_bytes(factor):
 return sum(t.numel()*t.element_size() for t in [factor['left'],*factor['rights']])

def dissimilar_edges(raw):
 layers=sorted(raw)
 if layers!=list(range(len(layers))):raise ValueError('Contiguous original layer IDs required')
 edges=[]
 for s in layers:
  for t in range(s+1,len(layers)):
   # For one immutable prefix, averaging is not over calibration prompts.
   a=(raw[s][0].float()+raw[s][1].float())*.5
   b=(raw[t][0].float()+raw[t][1].float())*.5
   if a.shape!=b.shape:raise ValueError('KV geometry mismatch')
   edges.append((float(torch.linalg.vector_norm(a-b)),s,t))
 return sorted(edges,key=lambda x:(-x[0],x[1],x[2]))

def kvsharer_style_search(edges,layers,target_sources,evaluate_cosine,*,threshold=.5,feasible=None):
 if not 1<=target_sources<=layers or not -1<=threshold<=1:raise ValueError('Bad search configuration')
 sources=list(range(layers));history=[]
 for distance,s,t in edges:
  if len(set(sources))<=target_sources:break
  if not (0<=s<t<layers) or not math.isfinite(distance):raise ValueError('Expected ascending layer directions')
  if sources[s]!=s or sources[t]!=t:continue
  # Keep one-hop ownership; do not silently change already accepted dependents.
  if any(i!=t and a==t for i,a in enumerate(sources)):continue
  proposed=sources.copy();proposed[t]=s
  if feasible and not feasible(proposed):
   history.append({'source':s,'consumer':t,'accepted':False,'reason':'prefix_support_contract'});continue
  sim=float(evaluate_cosine(proposed))
  if not math.isfinite(sim) or not -1.00001<=sim<=1.00001:raise ValueError('Invalid cosine result')
  accepted=sim>threshold
  history.append({'source':s,'consumer':t,'distance':distance,'cosine':sim,'accepted':accepted})
  if accepted:sources=proposed
 validate_sources(sources)
 return {'sources':sources,'target_sources':target_sources,'achieved_sources':len(set(sources)),
    'target_reached':len(set(sources))==target_sources,'history':history,
    'variant':'KVSharer-style fixed-prefix search reference; explicit adaptation, NOT an official full reproduction'}
