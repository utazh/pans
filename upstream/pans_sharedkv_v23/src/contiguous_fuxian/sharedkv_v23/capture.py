"""Capture full group boundaries AND actual teacher-forced label prefixes.

max_tokens must be 1, so subsequent decoder calls are the runner's label
continuations, not potentially divergent free generation. No gold labels are
read. Causal branches are reassembled as full sequences for group replay.
"""
from __future__ import annotations
from contextvars import ContextVar
from pathlib import Path
import torch
from .contracts import TRACE,write_new
from contiguous_fuxian.sharedkv.io import query_hash
from contiguous_fuxian.sharedkv_v21.traces import fingerprint

_ACTIVE=ContextVar('v23_capture',default=None)


def assemble(base,steps,max_suffix):
    paths=sorted(steps,key=lambda x:(len(x),x))
    leaves=[p for p in paths if len(p)<=max_suffix and not any(len(q)>len(p) and q[:len(p)]==p for q in paths if len(q)<=max_suffix)]
    examples=[]
    for path in [(),*leaves]:
        calls=[base]+[steps[path[:i]] for i in range(1,len(path)+1)]
        ex={'suffix_ids':list(path)}
        for key in ('layer_inputs','layer_outputs','attention_outputs'):
            ex[key]={lid:torch.cat([c[key][lid] for c in calls],dim=1) for lid in base[key]}
        examples.append(ex)
    return examples


class GroupCollector:
    def __init__(self,root,start,end,max_suffix=4):
        self.root=Path(root);self.start=start;self.end=end;self.max_suffix=max_suffix
        if not 0<=start<end or max_suffix<0:raise ValueError('Invalid capture range')

    def install(self,runtime,runner):
        old_prepare=runtime._prepare
        def prepare(*args,**kwargs):
            value=old_prepare(*args,**kwargs)
            # ONLY this explicit calibration collector waives evaluation bans.
            return (*value[:3],set())
        runtime._prepare=prepare
        old_decoder=runner.flexgen_sparse_decoder_logits
        def decoder(*args,**kwargs):
            state=_ACTIVE.get()
            if state is None:return old_decoder(*args,**kwargs)
            ids=kwargs['input_ids'][0].detach().cpu().tolist()
            pos=int(kwargs['position_start']);n=state['prefix_tokens'];T=len(state['query_token_ids'])
            if pos==n:
                if state['base'] is not None or ids!=state['query_token_ids']:
                    raise ValueError('Expected a single full original query prefill')
                path=()
            else:
                offset=pos-n-T
                if len(ids)!=1 or offset<0:raise ValueError('Unexpected continuation geometry')
                if offset==0:path=(ids[0],)
                else:
                    previous=state['last_path']
                    if len(previous)!=offset:raise ValueError('Cannot recover label continuation parent')
                    path=previous+(ids[0],)
            state['last_path']=path
            call={'layer_inputs':{},'layer_outputs':{},'attention_outputs':{},'pre_mlp':{}}
            state['current']=call
            try:result=old_decoder(*args,**kwargs)
            finally:state['current']=None
            if len(path)<=self.max_suffix:
                for field in ('layer_inputs','layer_outputs','attention_outputs'):
                    if len(call[field])!=self.end-self.start:raise RuntimeError('Capture missed group layers')
                    call[field]={k:v.detach().cpu().clone() for k,v in call[field].items()}
                del call['pre_mlp']
                if not path:state['base']=call
                elif path not in state['steps']:state['steps'][path]=call
                else:
                    # Repeated label-token prefixes must have the same state.
                    for k in call['layer_outputs']:
                        if not torch.equal(call['layer_outputs'][k],state['steps'][path]['layer_outputs'][k]):
                            raise RuntimeError('Identical label prefixes produced different capture states')
            return result
        runner.flexgen_sparse_decoder_logits=decoder
        old_completion=runner.greedy_flexgen_completion
        def completion(*args,**kwargs):
            if kwargs['max_tokens']!=1:raise ValueError('Capture requires max_tokens=1; labels are forced by original scorer')
            model,loader=kwargs['model'],kwargs['loader'];period=kwargs['impress_selection_period_size']
            g,sources,readers,_=runtime._prepare(model,loader,period)
            if self.end>g['layers'] or self.start//period!=(self.end-1)//period:
                raise ValueError('Capture must remain in one selector window')
            if any(l>=self.start for l in readers):
                raise ValueError('Capture student may only contain fixed upstream readers before this group')
            qids=[int(x) for x in kwargs['query_token_ids']];digest=query_hash(qids)
            state={'query_token_ids':qids,'prefix_tokens':loader.prefix_tokens,'base':None,'steps':{},'current':None,'last_path':()}
            handles=[]
            def before_norm(lid):
                def hook(_module,inputs):
                    c=state['current']
                    if c is not None:c['layer_inputs'][lid]=inputs[0].detach()
                return hook
            def before_post(lid):
                def hook(_module,inputs):
                    if state['current'] is not None:state['current']['pre_mlp'][lid]=inputs[0].detach()
                return hook
            def before_o(lid):
                def hook(_module,inputs):
                    if state['current'] is not None:state['current']['attention_outputs'][lid]=inputs[0].detach()
                return hook
            def after_mlp(lid):
                def hook(_module,_inputs,output):
                    if state['current'] is not None:
                        c=state['current'];c['layer_outputs'][lid]=c['pre_mlp'][lid]+output.detach()
                return hook
            for lid in range(self.start,self.end):
                layer=model.model.layers[lid]
                handles.extend([layer.input_layernorm.register_forward_pre_hook(before_norm(lid)),
                    layer.post_attention_layernorm.register_forward_pre_hook(before_post(lid)),
                    layer.self_attn.o_proj.register_forward_pre_hook(before_o(lid)),
                    layer.mlp.register_forward_hook(after_mlp(lid))])
            token=_ACTIVE.set(state)
            try:
                result=old_completion(*args,**kwargs)
                if state['base'] is None:raise RuntimeError('No primary prefill captured')
                row={'format':TRACE,'task':loader._info.task,'query_hash':digest,'query_token_ids':qids,
                    'token_hash':loader._info.token_hash,'prefix_tokens':loader.prefix_tokens,
                    'geometry':g,'period':period,'keep_ratio':float(loader.keep_ratio),
                    'model_identity':loader._selector_index.payload['model_identity'],
                    'group_start':self.start,'group_end':self.end,
                    'rollout_sources':sources,'rollout_reader_fingerprint':fingerprint(readers),
                    'selection_by_layer':[list(x) for x in loader._selected_by_layer],
                    'variants':assemble(state['base'],state['steps'],self.max_suffix),
                    'phase':'full query and forced label prefixes; not performance data'}
                write_new(self.root/row['task']/(digest+'.pt'),row)
                write_new(self.root/'selection'/(digest+'.pt'),{'selected_by_layer':row['selection_by_layer']})
                return result
            finally:
                _ACTIVE.reset(token)
                for h in handles:h.remove()
        runner.greedy_flexgen_completion=completion
