"""Export a fixed-source direct-sharing ablation; not an implementation of KVSharer."""
from __future__ import annotations
from pathlib import Path
import copy,hashlib
import torch
from contiguous_fuxian.sharedkv_v25.policy import FORMAT,NativePolicy

def make_direct(bundle):
    if bundle.get('format')!=FORMAT:raise ValueError('A standard v25 checkpoint is required')
    NativePolicy(**bundle['native_policy'])
    result=copy.deepcopy(bundle);g=bundle['geometry'];h,d=g['kv_heads'],g['head_dim']
    consumers={str(l) for l,s in enumerate(bundle['sources']) if l!=s}
    if set(bundle['readers'])!=consumers:raise ValueError('Reader/source contract mismatch')
    for key in consumers:
        old=bundle['readers'][key]
        dtype=old['A'].dtype
        result['readers'][key]={'A':torch.eye(d,dtype=dtype).repeat(h,1,1),
            'B':torch.eye(d,dtype=dtype).repeat(h,1,1),
            'bk':torch.zeros(h,d,dtype=dtype),'bv':torch.zeros(h,d,dtype=dtype)}
    result.update(diagnostic_only=True,accepted_consumers=0,end_to_end_validated=False,
        deployment_status='v26_fixed_structure_direct_control',compiler_family='v26_direct_ablation',
        control_note='Identity A/B and zero biases; SAME source map and native policy. No refitting. '
                     'Not the full source-selection method of any external baseline.')
    return result

def export_direct(source,output):
    source,output=Path(source),Path(output)
    bundle=torch.load(source,map_location='cpu',weights_only=True)
    result=make_direct(bundle)
    result['control_parent_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:torch.save(result,f)
