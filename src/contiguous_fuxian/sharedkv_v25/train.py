"""Use the repaired v24 trainer/loss/teacher contracts with v25 prefix partition.

No new score loss, no weaker numerical gate. All installed globals are restored.
A training candidate is never automatically certified or deployed.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
from .policy import NativePolicy,FORMAT,BASE

@contextmanager
def training_patch(policy):
    from contiguous_fuxian.sharedkv_v24 import train as t
    from .replay import selected_bank,continuation
    names=('selected_bank','continuation_tokens','initialize','save_new')
    old={name:getattr(t,name) for name in names}
    def bank(raw,row,sources,device,dtype):
        return selected_bank(raw,row,sources,device,dtype,policy)
    def initialize(raw,sources,initial,geometry,ridge,device):
        if initial is not None and initial.get('format')==FORMAT:
            NativePolicy(**initial['native_policy'])
            initial=copy.deepcopy(initial);initial['format']=t.BUNDLE_FORMAT
        return old['initialize'](raw,sources,initial,geometry,ridge,device)
    def save(path,data):
        if not isinstance(data,dict) or 'readers' not in data:
            return old['save_new'](path,data)
        data=copy.deepcopy(data)
        data.update(format=FORMAT,native_policy=policy.state(),runtime_required='sharedkv_v25',
                    audited_runtime_base=BASE,compiler_family='v25_budgeted_native_suffix',
                    diagnostic_only=True,deployment_status='exploratory_candidate_not_certified',
                    end_to_end_validated=False,accepted_consumers=0,
                    materialization_scope='Full owner selections plus partial native prefix patches; no full consumer KV reconstructed',
                    gate_note='v24 failed NI results remain unchanged; exploration is not certification')
        return old['save_new'](path,data)
    t.selected_bank=bank;t.continuation_tokens=continuation;t.initialize=initialize;t.save_new=save
    try:yield t
    finally:
        for name,value in old.items():setattr(t,name,value)
