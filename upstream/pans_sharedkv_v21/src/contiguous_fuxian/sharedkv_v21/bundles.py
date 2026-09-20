"""Isolation ablations preserving identity, calibration bans, and rejection status."""
import copy
import torch
from contiguous_fuxian.sharedkv.io import validate_sources

def isolate(bundle,targets,*,direct=False):
    targets=sorted(set(int(t) for t in targets))
    if not targets:raise ValueError('Specify at least one consumer')
    out=copy.deepcopy(bundle);sources=list(range(len(bundle['sources'])));readers={}
    for t in targets:
        if t<0 or t>=len(sources) or str(t) not in bundle['readers']:raise ValueError(f'No reader for consumer {t}')
        source=bundle['sources'][t]
        if source==t:raise ValueError('Requested native layer is not a consumer')
        sources[t]=source;r=copy.deepcopy(bundle['readers'][str(t)])
        if direct:
            h,d,_=r['A'].shape
            r={ 'A':torch.eye(d,dtype=r['A'].dtype).repeat(h,1,1),
                'B':torch.eye(d,dtype=r['B'].dtype).repeat(h,1,1),
                'bk':torch.zeros_like(r['bk']),'bv':torch.zeros_like(r['bv']) }
        readers[str(t)]=r
    validate_sources(sources,period=bundle['period'])
    out.update(sources=sources,readers=readers,diagnostic_only=True,
        deployment_status='diagnostic_candidate_not_accepted',end_to_end_validated=False,
        accepted_consumers=0,candidate_consumers=len(readers),local_gate_passed=False,
        ablation={'targets':targets,'direct':direct,'parent_sources':bundle['sources']})
    # Parent local reports no longer certify this bundle.
    out['parent_report_for_provenance']=out.pop('report',None)
    return out
