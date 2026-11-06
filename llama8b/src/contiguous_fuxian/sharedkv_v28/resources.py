"""Representation arithmetic on FIT supports; not hardware I/O or peak memory."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .common import load_tensor

def reference_bytes(bundle, plan):
    if bundle['method']=='direct-map-prefix-quality':
        unions={}
        for lid,src in enumerate(bundle['sources']):
            unions.setdefault(src,set()).update(plan[lid])
        # Quality bundles hold FP32 copies; a proposed BF16 payload is a distinct
        # hypothetical deployment, so use the ACTUAL tensor element sizes here.
        restored=sum(len(ids)*sum(x.shape[1]*x.shape[2]*x.element_size()
                     for x in bundle['native'][str(src)]) for src,ids in unions.items())
        resident=sum(x.numel()*x.element_size() for kv in bundle['native'].values() for x in kv)
        return dict(representation_tensor_bytes=resident,selected_rows_reference_bytes=restored,
                    rights_resident_bytes=0,source_union_tokens=sum(map(len,unions.values())))
    if bundle['method']!='xKV-CLF-prefix-quality':raise ValueError('Unsupported reference representation')
    resident=rows=rights=0
    for group in bundle['groups']:
        union=set().union(*(set(plan[lid]) for lid in group['layers']))
        for kind in ('key','value'):
            fac=group[kind];left=fac['left']
            resident+=left.numel()*left.element_size()
            rows+=len(union)*fac['rank']*left.element_size()
            size=sum(x.numel()*x.element_size() for x in fac['rights'])
            resident+=size;rights+=size
    return dict(representation_tensor_bytes=resident,selected_rows_reference_bytes=rows,
                rights_resident_bytes=rights,source_union_tokens=None)

def collect(m):
    from contiguous_fuxian.sharedkv_v24.records import load
    out=[];root=Path(m['root'])
    for c in m['cells']:
        paths=sorted((root/'representations'/c['id']).glob('*.pt'))
        if not paths:continue
        fit=load(c['teacher_fit'],c['task'],'fit')
        for path in paths:
            if path.stem in ('direct_identity','clf_full64'):continue
            bundle=load_tensor(path)
            values=[reference_bytes(bundle,r['selection_by_layer']) for r in fit.values()]
            result={k:(float(np.mean([v[k] for v in values])) if values[0][k] is not None else None)
                    for k in values[0]}
            out.append(dict(cell=c['id'],arm=path.stem,**result,
                 evidence_kind='theoretical_tensor_accounting_on_teacher_FIT_supports',
                 excludes='reconstruction workspace, private KV, index, allocator, file read amplification',
                 hardware_io_valid=False,latency_valid=False,
                 note='Quality bridge also restores native payload. Do not equate this arithmetic with its actual I/O.'))
    return out
