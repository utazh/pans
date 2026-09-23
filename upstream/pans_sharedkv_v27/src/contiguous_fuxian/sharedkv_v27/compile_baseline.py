"""Compile xKV's CLF mathematical reference, or an externally selected direct map.

No online factors-I/O implementation is implied. Full-rank reconstruction must
be validated against identity before rank experiments. Qwen full fixed RoPE only.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from .common import FIELDS,load_tensor,torch_new,validate_sources,digest
from .baseline_math import factorize,rotary,factor_bytes
from .baseline_quality import FORMAT

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--method',choices=['clf','direct-map'],required=True)
 for k in ('teacher-fit','store-root','task','model-config','output'):p.add_argument('--'+k,required=True)
 p.add_argument('--window',type=int,default=4);p.add_argument('--rank-k',type=int,default=256);p.add_argument('--rank-v',type=int,default=384)
 p.add_argument('--direct-plan');p.add_argument('--seed',type=int,default=27)
 p.add_argument('--device',default='cpu');p.add_argument('--exact-svd',action='store_true')
 a=p.parse_args(argv)
 if Path(a.output).exists():raise FileExistsError(a.output)
 from contiguous_fuxian.sharedkv_v24.records import load
 from contiguous_fuxian.sharedkv.io import read_prefix
 rows=load(a.teacher_fit,a.task,'fit');head=next(iter(rows.values()))
 cfg=json.loads(Path(a.model_config).read_text());g=head['geometry']
 if cfg.get('model_type')!='qwen2' or cfg.get('rope_scaling') is not None or cfg.get('use_sliding_window',False):
  raise ValueError('Only full fixed-RoPE Qwen2; no dynamic/partial scaling')
 geom={'layers':cfg['num_hidden_layers'],'q_heads':cfg['num_attention_heads'],'kv_heads':cfg['num_key_value_heads'],
   'head_dim':cfg.get('head_dim',cfg['hidden_size']//cfg['num_attention_heads'])}
 if g!=geom:raise ValueError('Config and teacher geometries differ')
 device=torch.device(a.device);n=head['prefix_tokens'];start=time.perf_counter();raw={}
 for lid in range(g['layers']):
  kv=[]
  for kind in ('key','value'):
   x,meta=read_prefix(a.store_root,a.task,lid,kind)
   if meta['token_hash']!=head['token_hash']:raise ValueError('Wrong prefix')
   kv.append(x.to(device=device,dtype=torch.float32))
  raw[lid]=kv
 out={'format':FORMAT,**{k:head[k] for k in FIELDS},'calibration_query_hashes':sorted(rows),
  'method':'xKV-CLF-prefix-quality' if a.method=='clf' else 'direct-map-prefix-quality',
  'quality_only':True,'latency_valid':False,'hardware_bytes_valid':False,
  'position_offset':0,'rope_theta':float(cfg.get('rope_theta',10000.)),
  'model_config_digest':digest(cfg),'compile_args':vars(a)}
 if a.method=='direct-map':
  if not a.direct_plan:raise ValueError('--direct-plan is required; this compiler does NOT search KVSharer')
  plan=json.loads(Path(a.direct_plan).read_text());sources=plan['sources'];validate_sources(sources)
  if len(sources)!=g['layers']:raise ValueError('Map length')
  if not plan.get('design_query_hashes') or not set(plan['design_query_hashes'])<=set(rows):
   raise ValueError('An explicit nonempty FIT-only source-map provenance is required')
  out.update(sources=sources,native={str(s):[x.cpu() for x in raw[s]] for s in sorted(set(sources))},
             map_provenance=plan)
 else:
  if a.window<1:raise ValueError('Invalid window')
  pos=torch.arange(n,device=device);groups=[];lookup=[]
  for lo in range(0,g['layers'],a.window):
   layers=list(range(lo,min(g['layers'],lo+a.window)))
   # Already-cached BF16 values were rounded in v25; inverse-RoPE is explicit.
   # This reference does not claim direct access to original pre-RoPE activations.
   ks=[rotary(raw[l][0],pos,theta=out['rope_theta'],inverse=True) for l in layers]
   vs=[raw[l][1] for l in layers]
   cap=min(n,len(layers)*g['kv_heads']*g['head_dim'])
   fk=factorize(ks,min(a.rank_k,cap),seed=a.seed,exact=a.exact_svd)
   fv=factorize(vs,min(a.rank_v,cap),seed=a.seed,exact=a.exact_svd)
   def cpu(f):return {**f,'left':f['left'].cpu(),'rights':[x.cpu() for x in f['rights']]}
   groups.append({'layers':layers,'key':cpu(fk),'value':cpu(fv)})
   lookup.extend([[len(groups)-1,i] for i in range(len(layers))])
  out.update(groups=groups,lookup=lookup,factor_bytes=sum(factor_bytes(g['key'])+factor_bytes(g['value']) for g in groups),
      pre_rope_note='Inverse of the observed fixed-RoPE rounded cache; validate full-rank roundtrip, not an official activation dump',
      selection_note='P8 online IDs, not xKV-SR landmark selector',official_system_reproduced=False)
 out['compile_seconds']=time.perf_counter()-start
 torch_new(a.output,out)
 print(json.dumps({k:out.get(k) for k in ('method','factor_bytes','compile_seconds','quality_only')},indent=2))
