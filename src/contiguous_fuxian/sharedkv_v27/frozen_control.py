"""Deployable compute path, UNVALIDATED model: frozen R4 weights -> R0 policy.

No optimization; no native-file/proxy read in R0. Changing the policy is an
explicit intervention, never a reinterpretation of the published checkpoint.
"""
from __future__ import annotations
import copy
from .common import V25_FORMAT,validate_sources,load_tensor,torch_new,sha_file

def make_frozen_r0(bundle):
 if bundle.get('format')!=V25_FORMAT:raise ValueError('Expected an audited v25 checkpoint')
 policy=bundle.get('native_policy',{})
 if policy.get('rule')!='last_selected_blocks' or type(policy.get('blocks')) is not int or policy['blocks']<=0:
  raise ValueError('Requires an R>0 last-selected-block checkpoint')
 if bundle.get('head_maps'):raise ValueError('Head routing is not part of this control')
 g=bundle['geometry'];sources=bundle['sources'];validate_sources(sources,bundle['period'])
 if len(sources)!=g['layers']:raise ValueError('Wrong layer count')
 targets={str(i) for i,s in enumerate(sources) if i!=s}
 if not targets or set(bundle['readers'])!=targets:raise ValueError('Reader/source mismatch')
 for r in bundle['readers'].values():
  for name,shape in [('A',(g['kv_heads'],g['head_dim'],g['head_dim'])),('B',(g['kv_heads'],g['head_dim'],g['head_dim'])),('bk',(g['kv_heads'],g['head_dim'])),('bv',(g['kv_heads'],g['head_dim']))]:
   import torch
   if tuple(r[name].shape)!=shape or not torch.isfinite(r[name]).all():raise ValueError('Bad reader')
 out=copy.deepcopy(bundle)
 out['native_policy']['blocks']=0
 # Do not retain an R4 performance/quality report as if it described this model.
 out['parent_quality_report']=out.pop('report',None)
 out.update(compiler_family='v27_frozen_r4_to_r0',diagnostic_only=True,
   deployment_status='frozen_policy_intervention_unvalidated',end_to_end_validated=False,
   quality_certified=False,local_gate_passed=False,accepted_consumers=0,
   intervention={'kind':'fixed_reader_no_native_patch','from_blocks':policy['blocks'],
   'reader_values_changed':False,'training_steps':0,
   'warning':'Not retrained R0; real arithmetic proxy equivalence does not guarantee BF16/logits/accuracy parity.'})
 return out

def export_frozen_r0(source,output):
 out=make_frozen_r0(load_tensor(source));out['control_parent_sha256']=sha_file(source)
 torch_new(output,out);return out
