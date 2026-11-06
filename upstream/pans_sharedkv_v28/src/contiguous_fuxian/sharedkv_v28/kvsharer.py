"""Parameterization of scripts/v27_search_kvsharer.py, NOT a new search rule.

Separate K/V mean token/head vector L2; descending distance; author's remap;
mean final-normalized-hidden cosine > .5, 30 sorted FIT queries, full query.
Prefix adaptation keeps private KV native and may use sources across P8 windows.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from pathlib import Path
from .common import FIELDS,save_tensor,write_json

def author_edges(raw):
 L=len(raw)
 if set(raw)!=set(range(L)):raise ValueError('Contiguous layers required')
 edges=[]
 for s in range(L):
  for t in range(s+1,L):
   distance=sum(float(torch.linalg.vector_norm(raw[s][j].float()-raw[t][j].float(),dim=-1).mean())
                for j in (0,1))/2
   edges.append((distance,s,t))
 return sorted(edges,key=lambda x:(-x[0],x[2],x[1]))
def author_proposal(sources,s,t):
 if not 0<=s<t<len(sources):raise ValueError('Expected earlier-source direction')
 proposed=list(sources);proposed[t]=s
 for i in range(len(proposed)):
  if proposed[i]!=i:proposed[i]=proposed[proposed[i]]
 if any(proposed[proposed[i]]!=proposed[i] for i in range(len(proposed))):
  raise ValueError('Author remap did not produce one-hop owners')
 return proposed

def compile_cell(c,settings,folder):
 from transformers import AutoModelForCausalLM
 from contiguous_fuxian.sharedkv.io import read_prefix,geometry
 from contiguous_fuxian.sharedkv_v24.records import load
 from contiguous_fuxian.sharedkv_v24.faithful import decoder
 from contiguous_fuxian.sharedkv_v27.baseline_quality import FORMAT
 from contiguous_fuxian.quantized_key_index import model_identity
 fit=load(c['teacher_fit'],c['task'],'fit');hashes=sorted(fit)[:30]
 if len(hashes)!=30:raise ValueError('Exactly 30 sorted FIT queries required, no evaluation fallback')
 header=fit[hashes[0]];L=header['geometry']['layers'];device=settings.get('search_device','cuda')
 if model_identity(settings['model_path'])!=header['model_identity']:raise ValueError('Wrong search backbone')
 raw={}
 for lid in range(L):
  kv=[]
  for k in ('key','value'):
   x,m=read_prefix(settings['store_root'],c['task'],lid,k)
   if m['token_hash']!=header['token_hash']:raise ValueError('Search prefix mismatch')
   kv.append(x.to(device=device,dtype=torch.bfloat16))
  raw[lid]=kv
 edges=author_edges(raw)
 model=AutoModelForCausalLM.from_pretrained(settings['model_path'],local_files_only=True,
          torch_dtype=torch.bfloat16,attn_implementation='eager',trust_remote_code=False).to(device).eval()
 model.requires_grad_(False)
 if geometry(model.config)!=header['geometry'] or model.config.model_type!='qwen2':raise ValueError('Unsupported search geometry')
 captured=[];hook=model.model.norm.register_forward_hook(lambda m,a,o:captured.append(o.detach()))
 native=list(range(L));baseline={};history=[];snapshots={};sources=native.copy();accepted=set()
 desired=set(settings.get('kvsharer_g',[24,20,16]));out=Path(folder)/'representations'/c['id']
 def evaluate(key,source_map):
  row=fit[key];bank={}
  for lid,s in enumerate(source_map):
   ids=torch.tensor(row['selection_by_layer'][lid],device=device,dtype=torch.long)
   bank[lid]=tuple(x.index_select(0,ids).unsqueeze(0) for x in raw[s])
  captured.clear()
  logits,_=decoder(model,torch.tensor([row['query_token_ids']],device=device),row['prefix_tokens'],
                   None,native,bank,{},False)
  hidden=captured[-1].float().flatten().clone();lp=logits[0,-1].float().log_softmax(-1)
  error=max(abs(float(lp[row['label_ids'][label][0]])-row['teacher_token_logprobs'][label][0])
            for label in row['label_order'])
  return hidden,error
 try:
  with torch.no_grad():
   parity=[]
   for key in hashes:
    hidden,error=evaluate(key,native);baseline[key]=hidden;parity.append(error)
   if max(parity)>=.02:
    write_json(out/'kvsharer_rejected_parity.json',dict(max_error=max(parity),fit_hashes=hashes))
    raise RuntimeError('Search native replay failed original .02 gate')
   for distance,s,t in edges:
    if t in accepted:continue
    proposed=author_proposal(sources,s,t)
    cosine=[float(F.cosine_similarity(evaluate(key,proposed)[0][None],baseline[key][None])) for key in hashes]
    mean=sum(cosine)/len(cosine);ok=mean>.5
    history.append(dict(source=s,consumer=t,distance=distance,cosine=mean,min_cosine=min(cosine),accepted=ok))
    print(f"{c['id']} KVSharer {s}->{t} cosine={mean:.6f} accepted={ok}",flush=True)
    if ok:
     sources=proposed;accepted.add(t);G=len(set(sources))
     if G in desired and G not in snapshots:
      snapshots[G]=list(sources)
      b=dict(format=FORMAT,**{k:header[k] for k in FIELDS},calibration_query_hashes=sorted(fit),
         method='direct-map-prefix-quality',quality_only=True,latency_valid=False,hardware_bytes_valid=False,
         position_offset=0,rope_theta=float(model.config.rope_theta),sources=list(sources),
         native={str(a):[x.cpu().float() for x in raw[a]] for a in sorted(set(sources))},
         map_provenance=dict(design_query_hashes=hashes,threshold=.5,history=history.copy(),
          variant='v27 author-distance/remap prefix adaptation; not official Llama/InternLM reproduction'))
      save_tensor(out/f'kvsharer_g{G}.pt',b)
     if G<=min(desired):break
 finally:hook.remove()
 write_json(out/'kvsharer_search.json',dict(fit_hashes=hashes,native_first_token_error=max(parity),
      desired_sources=sorted(desired),achieved_sources=sorted(snapshots),unreached=sorted(desired-set(snapshots)),
      history=history,failed_targets_not_filled=True))
 # Export a native identity representation for the unchanged quality bridge gate.
 save_tensor(out/'direct_identity.pt',dict(format=FORMAT,**{k:header[k] for k in FIELDS},
    calibration_query_hashes=sorted(fit),method='direct-map-prefix-quality',quality_only=True,latency_valid=False,
    hardware_bytes_valid=False,position_offset=0,rope_theta=float(model.config.rope_theta),sources=native,
    native={str(i):[x.cpu().float() for x in raw[i]] for i in native}))
