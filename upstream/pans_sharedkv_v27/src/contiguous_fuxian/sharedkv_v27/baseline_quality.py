"""End-to-end QUALITY bridge for mathematical prefix baselines.

Uses actual online Qwen/P8/private-cache path, but ALSO restores all original
payload via the unchanged loader, then replaces values for attention. Factors
are resident. Thus latencies / loader byte counts MUST NEVER be baseline
performance results. Every run requires --diagnostic and emits ineligible tags.

This separates representation/quality validation from the still-required
SSD-native external baseline integration. It does NOT claim an xKV-SR engine.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch
from .common import FIELDS,load_tensor,validate_sources
from .baseline_math import reconstruct,rotary

FORMAT='pans-v27-prefix-quality-baseline'

class QualityMemory:
 def __init__(self,bundle):
  if bundle.get('format')!=FORMAT:raise ValueError('Not a quality baseline bundle')
  self.bundle=bundle;self.checked=set();self.ready={}
 def prepare(self,loader,device):
  b=self.bundle
  expected={'task':loader._info.task,'token_hash':loader._info.token_hash,
    'prefix_tokens':loader.prefix_tokens,'keep_ratio':float(loader.keep_ratio)}
  for k,v in expected.items():
   if b[k]!=v:raise ValueError(f'Baseline identity mismatch: {k}')
  if loader._selector_index is None or b['model_identity']!=loader._selector_index.payload['model_identity']:
   raise ValueError('Wrong model/index identity')
  if loader.layers!=b['geometry']['layers']:raise ValueError('Wrong layer count')
  key=str(device)
  if key not in self.ready:
   def move(x):
    if torch.is_tensor(x):return x.to(device)
    if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
    if isinstance(x,list):return [move(v) for v in x]
    return x
   self.ready[key]=move(b)
  return self.ready[key]
 def get(self,loader,layer,ids,device,dtype):
  b=self.prepare(loader,device);idx=torch.tensor(ids,device=device,dtype=torch.long)
  if b['method']=='direct-map-prefix-quality':
   source=b['sources'][layer]
   return tuple(x.index_select(0,idx).to(dtype).unsqueeze(0) for x in b['native'][str(source)])
  if b['method']!='xKV-CLF-prefix-quality':raise ValueError('Unknown representation')
  group_id,local=b['lookup'][layer];group=b['groups'][group_id]
  k=reconstruct(group['key'],local,idx)
  k=rotary(k,idx+b['position_offset'],theta=b['rope_theta'])
  v=reconstruct(group['value'],local,idx)
  return k.to(dtype).unsqueeze(0),v.to(dtype).unsqueeze(0)

@contextmanager
def install_quality_memory(bundle):
 from contiguous_fuxian.sharedkv.runtime import RequestState
 original=RequestState.resolve;memory=QualityMemory(bundle)
 def resolve(state,loader,layer,*,dtype,device):
  if state.readers or state.sources!=list(range(len(state.sources))):raise ValueError('Use --mode identity, not learned sharing')
  # Still execute original loader: valid numerical/context baseline, invalid I/O timing.
  if not hasattr(state,'_v27_quality_done'):state._v27_quality_done=set()
  if layer in state._v27_quality_done:return state.bank[layer]
  original(state,loader,layer,dtype=dtype,device=device)
  kv=memory.get(loader,layer,loader._selected_by_layer[layer],device,dtype)
  state.bank[layer]=kv;loader._loaded[layer]=tuple(x[0] for x in kv)
  state._v27_quality_done.add(layer)
  return kv
 RequestState.resolve=resolve
 try:yield memory
 finally:RequestState.resolve=original
