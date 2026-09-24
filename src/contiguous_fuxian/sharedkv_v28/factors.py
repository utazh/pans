"""One exact FP64 master; same U / (S Vh) scaling for BOTH FP32 and BF16.

Exports are compatible with the existing v27 QUALITY bridge only. No SSD I/O
engine, landmark selector, compressed serving cache, or speed claim is provided.
"""
from __future__ import annotations
import copy,time
from pathlib import Path
import torch
from .common import FIELDS,save_tensor,load_tensor,write_json,sha,compare_identity

def svd_master(tensors):
 if not tensors or any(t.shape!=tensors[0].shape or t.ndim!=3 for t in tensors):
  raise ValueError('Equal native [N,H,D] tensors required')
 n,h,d=tensors[0].shape;x=torch.cat([t.double().reshape(n,h*d) for t in tensors],1)
 kwargs={'driver':'gesvd'} if x.device.type=='cuda' else {}
 u,s,vh=torch.linalg.svd(x,full_matrices=False,**kwargs)
 # Store full-rank master. Zero singular values are safe: never divide by S.
 return {'u':u.cpu().contiguous(),'s':s.cpu().contiguous(),'vh':vh.cpu().contiguous(),
         'geometry':[n,h,d],'layers':len(tensors),'solver':'exact_fp64_gesvd' if kwargs else 'exact_fp64_lapack'}
def export_factor(master,rank,dtype):
 cap=len(master['s'])
 if not 1<=rank<=cap:raise ValueError(f'Rank {rank} exceeds cap {cap}; no silent clipping')
 n,h,d=master['geometry'];vh=master['vh'][:rank]
 svh=master['s'][:rank,None]*vh
 rights=[svh[:,i*h*d:(i+1)*h*d].to(dtype).contiguous() for i in range(master['layers'])]
 return dict(left=master['u'][:,:rank].to(dtype).contiguous(),rights=rights,
             geometry=[n,h,d],rank=rank)
def make_master(header,store_root,config,output,window=4,device='cpu'):
 from contiguous_fuxian.sharedkv.io import read_prefix
 from contiguous_fuxian.sharedkv_v27.baseline_math import rotary
 if window<=0:raise ValueError('Invalid window')
 g=header['geometry'];n=header['prefix_tokens'];groups=[];lookup=[];start=time.perf_counter()
 if config['model_type']!='qwen2' or config.get('rope_scaling') is not None or config.get('use_sliding_window',False):
  raise ValueError('Fixed full-RoPE Qwen2 only')
 geom=dict(layers=config['num_hidden_layers'],q_heads=config['num_attention_heads'],
           kv_heads=config['num_key_value_heads'],head_dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads']))
 if geom!=g:raise ValueError('Master geometry mismatch')
 theta=float(config.get('rope_theta',10000.));pos=torch.arange(n,device=device)
 for lo in range(0,g['layers'],window):
  layers=list(range(lo,min(lo+window,g['layers'])));kv={'key':[],'value':[]}
  for lid in layers:
   for kind in kv:
    x,m=read_prefix(store_root,header['task'],lid,kind)
    if m['token_hash']!=header['token_hash']:raise ValueError('Prefix identity mismatch')
    x=x.to(device=device,dtype=torch.float64)
    if kind=='key':x=rotary(x,pos,theta=theta,inverse=True)
    kv[kind].append(x)
  group={'layers':layers,**{k:svd_master(v) for k,v in kv.items()}};groups.append(group)
  lookup.extend([[len(groups)-1,i] for i in range(len(layers))])
  print(f"FP64 master {header['task']} layers={layers}",flush=True)
 result=dict(format='pans-fp64-master-v28',task=header['task'],token_hash=header['token_hash'],
             model_identity=header['model_identity'],geometry=g,prefix_tokens=n,rope_theta=theta,
             window=window,groups=groups,lookup=lookup,compile_seconds=time.perf_counter()-start)
 save_tensor(output,result);return result

def representation(master,header,fit_hashes,ranks=None,dtype=torch.float64):
 from contiguous_fuxian.sharedkv_v27.baseline_quality import FORMAT
 from contiguous_fuxian.sharedkv_v27.baseline_math import factor_bytes
 for key in ('task','token_hash','model_identity','geometry','prefix_tokens'):
  if master[key]!=header[key]:raise ValueError('Master/checkpoint mismatch: '+key)
 groups=[]
 for group in master['groups']:
  item={'layers':list(group['layers'])}
  for i,kind in enumerate(('key','value')):
   rank=len(group[kind]['s']) if ranks is None else ranks[i]
   item[kind]=export_factor(group[kind],rank,dtype)
  groups.append(item)
 return dict(format=FORMAT,**{k:header[k] for k in FIELDS},calibration_query_hashes=sorted(fit_hashes),
   method='xKV-CLF-prefix-quality',quality_only=True,latency_valid=False,hardware_bytes_valid=False,
   position_offset=0,rope_theta=master['rope_theta'],groups=groups,lookup=master['lookup'],
   factor_bytes=sum(factor_bytes(g[k]) for g in groups for k in ('key','value')),
   factor_layout='U / S*Vh, exact common FP64 master; only rank/cast differs',
   factor_precision=str(dtype),full_rank_reference=ranks is None,
   variant='xKV-CLF/P8 prefix adaptation; not official xKV-SR or SSD-native factor restoration')

def compile_cell(c,settings,folder):
 from contiguous_fuxian.sharedkv_v24.records import load
 import json
 fit=load(c['teacher_fit'],c['task'],'fit');header=next(iter(fit.values()))
 cache=Path(folder)/'masters'/f"{c['task']}_{header['token_hash'][:16]}.pt"
 if cache.exists():master=load_tensor(cache)
 else:master=make_master(header,settings['store_root'],json.loads((Path(settings['model_path'])/'config.json').read_text()),
                         cache,window=4,device=settings.get('svd_device','cpu'))
 out=Path(folder)/'representations'/c['id'];out.mkdir(parents=True,exist_ok=True)
 names=[]
 for rk,rv in settings.get('ranks',[[256,384],[384,576]]):
  for name,dtype in [('fp32',torch.float32),('bf16',torch.bfloat16)]:
   tag=f'clf_{rk}_{rv}_{name}';p=out/(tag+'.pt')
   b=representation(master,header,fit,ranks=(rk,rv),dtype=dtype)
   b['master_sha256']=sha(cache);save_tensor(p,b);names.append(tag)
 p=out/'clf_full64.pt';b=representation(master,header,fit);b['master_sha256']=sha(cache);save_tensor(p,b)
 write_json(out/'clf_compile.json',dict(master=str(cache),master_sha256=sha(cache),artifacts=names,
             note='Full-rank runtime parity is a separate REQUIRED job. No gate relaxed.'))
