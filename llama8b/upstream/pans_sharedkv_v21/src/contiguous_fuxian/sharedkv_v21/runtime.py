"""Additive integration audited against utazh/pans v1 @ 59312f2.

Uses published SharedRuntime (including diagnostic gate, alias stats, tail forks).
Fixed plans STILL execute the selector: only configured IDs are overwritten.
No trace/profiler/redundant-load run is eligible as final end-to-end evidence.
"""
from __future__ import annotations
from contextvars import ContextVar
from contextlib import nullcontext
from pathlib import Path
import json
import torch
from contiguous_fuxian.sharedkv import runtime as v1
from contiguous_fuxian.sharedkv.core import apply_rope
from contiguous_fuxian.sharedkv.io import query_hash,load_safe
from .ops import pack,dispatch,nvtx
from .backend import LengthPool,_ACTIVE_POOL
from .traces import TraceSink,load_traces,fingerprint
_SETTINGS=ContextVar('sharedkv_v21_settings',default=None)


def schedule_window(state,loader,start,end,policy='window',shadow=False):
    if policy not in ('window','demand'): raise ValueError('Unknown scheduling policy')
    for lid in range(start,end):
        if loader._selected_by_layer[lid]!=loader._selected_by_layer[state.sources[lid]]:
            raise ValueError('Same IDs required; no hidden union loading')
    if policy=='window':
        state.schedule_window(loader,start,end)
        if shadow:
            for lid in range(start,end):
                if state.sources[lid]!=lid: loader.schedule(lid)
    # In demand mode, published loader.resolve schedules on first consumption.


def shadow_resolve(state,loader,lid,cfg,dtype,device):
    if not cfg.get('shadow_load',False) or state.sources[lid]==lid or lid in cfg['shadow_buffers']:
        return
    k,v=loader.resolve(lid)
    # Diagnostic counterfactual: restore independent consumer bytes but NEVER use
    # them for attention. Retain buffers to model their lifetime, not free early.
    cfg['shadow_buffers'][lid]=(k.to(device=device,dtype=dtype).contiguous(),
                                v.to(device=device,dtype=dtype).contiguous())


def decoder_logits(*,model,input_ids,cache,position_start,loader,period_size,subperiod_size,
                   impress_selection_block_size=1,impress_period_prefetch_size=1,
                   impress_selection_period_size=1,impress_known_period_prefetch=False):
    state,cfg=v1._ACTIVE.get(),_SETTINGS.get()
    if state is None or cfg is None: raise RuntimeError('Missing request context')
    cfg['state']=state
    core=model.model; hidden=core.embed_tokens(input_ids); b,t,_=hidden.shape
    if b!=1: raise ValueError('Only batch=1 supported')
    positions=torch.arange(position_start,position_start+t,device=hidden.device)
    cos,sin=core.rotary_emb(hidden,positions.unsqueeze(0))
    first=cache.get_seq_length(0)==0
    from contiguous_fuxian.flexgen_qwen_reprefill import configure_online_layer_selection
    from contiguous_fuxian.sharedkv.selection import fixed_selection
    g=state.geometry
    for lid,layer in enumerate(core.layers):
        if first and lid%impress_selection_period_size==0:
            end=min(g['layers'],lid+impress_selection_period_size)
            # All scoring/index transfer still happens. Replay changes only IDs.
            replay=cfg.get('replay')
            scope=fixed_selection(loader,replay['selection_by_layer']) if replay is not None else nullcontext()
            with nvtx(f'sharedkv/selector_{lid:02d}',cfg['nvtx']),scope:
                period=configure_online_layer_selection(decoder_layer=layer,hidden_states=hidden,
                    position_embeddings=(cos,sin),layer_index=lid,loader=loader,period_size=period_size,
                    impress_selection_block_size=impress_selection_block_size,
                    impress_selection_period_size=impress_selection_period_size)
                if period!=impress_selection_period_size: raise ValueError('Adaptive period unsupported')
                cfg['selector_calls']+=1
            with nvtx('sharedkv/schedule',cfg['nvtx']):
                schedule_window(state,loader,lid,end,cfg.get('schedule','window'),cfg.get('shadow_load',False))
                loader.prefetch_selector_keys(end)
        with nvtx(f'sharedkv/resolve_{lid:02d}',cfg['nvtx']):
            pk,pv=state.resolve(loader,lid,dtype=hidden.dtype,device=hidden.device)
            shadow_resolve(state,loader,lid,cfg,hidden.dtype,hidden.device)
        residual=hidden
        with nvtx('sharedkv/projection_rope_private_cache',cfg['nvtx']):
            x=layer.input_layernorm(hidden); attn=layer.self_attn
            q=attn.q_proj(x).view(b,t,g['q_heads'],g['head_dim'])
            k=attn.k_proj(x).view(b,t,g['kv_heads'],g['head_dim'])
            value=attn.v_proj(x).view(b,t,g['kv_heads'],g['head_dim'])
            q,k=apply_rope(q,cos,sin),apply_rope(k,cos,sin)
            tk,tv=cache.update(k.transpose(1,2),value.transpose(1,2),lid,
                              {'cos':cos,'sin':sin,'cache_position':positions})
            state.tail_lengths[lid]=max(state.tail_lengths.get(lid,0),int(tk.shape[2]))
            tk,tv=tk.transpose(1,2),tv.transpose(1,2)
        with nvtx(f'sharedkv/attention_{lid:02d}',cfg['nvtx']):
            out=dispatch(q,pk,pv,tk,tv,reader=state.readers.get(lid),
                packed=cfg['packed'].get(lid) if cfg['packed_ops'] else None,
                backend=state.backend,layout=cfg['layout'],annotate=cfg['nvtx'])
        with nvtx('sharedkv/output_projection_mlp',cfg['nvtx']):
            pre_mlp=residual+attn.o_proj(out.reshape(b,t,-1).contiguous())
            hidden=pre_mlp+layer.mlp(layer.post_attention_layernorm(pre_mlp))
        if first and cfg['sink'] is not None:
            cfg['sink'].record(lid,q=q,tk=tk,tv=tv,hidden_in=residual,pre_mlp=pre_mlp,
                hidden_out=hidden,out=out,selected=loader._selected_by_layer[lid],layer=layer)
    return model.lm_head(core.norm(hidden))


def storage_bytes(tensors):
    storages={}
    for x in tensors:
        s=x.untyped_storage(); storages[(str(x.device),s.data_ptr())]=s.nbytes()
    return sum(storages.values())


class Runtime(v1.SharedRuntime):
    def __init__(self,*,mode,backend,adapter_dir=None,stats_path=None,layout='split',packed_ops=False,
                 diagnostic=False,allow_candidate=False,trace_dir=None,replay_dir=None,trace_layers=None,
                 max_samples=8,annotate=False,schedule='window',metadata_cache=False,shadow_load=False,selection_plan_in_use=False):
        super().__init__(mode=mode,backend=backend,adapter_dir=adapter_dir,stats_path=stats_path,
                         allow_diagnostic=bool(diagnostic or allow_candidate))
        if (trace_dir or replay_dir or annotate or shadow_load) and not diagnostic:
            raise ValueError('trace/replay/NVTX/shadow_load requires --diagnostic')
        if layout not in ('split','hybrid') or schedule not in ('window','demand'):
            raise ValueError('Invalid execution configuration')
        self.layout,self.packed_ops=layout,packed_ops
        self.diagnostic,self.allow_candidate=diagnostic,allow_candidate
        self.trace_dir,self.replay_dir,self.trace_layers=trace_dir,replay_dir,trace_layers
        self.max_samples,self.annotate=max_samples,annotate
        self.schedule,self.metadata_cache,self.shadow_load=schedule,metadata_cache,shadow_load
        self.selection_plan_in_use=selection_plan_in_use
        self.packed={}; self.replays={}; self.metadata={}; self.identities={}

    def _prepare(self,model,loader,period):
        task=loader._info.task
        identity=(id(model),loader._info.token_hash,loader.prefix_tokens,float(loader.keep_ratio),period)
        if task in self.identities and self.identities[task]!=identity:
            raise ValueError('A runtime cannot silently reuse prepared readers across configurations')
        if task not in self.prepared and self.mode=='shared':
            bundle=load_safe(self.adapter_dir/f'{task}.pt')
            if bundle.get('deployment_status')!='independent_dev_passed' and not (self.diagnostic or self.allow_candidate):
                raise ValueError('Uncertified bundle requires --allow-candidate (held-out validation) or --diagnostic')
        result=super()._prepare(model,loader,period)
        if task not in self.packed:
            self.packed[task]={lid:pack(r) for lid,r in result[2].items()} if self.packed_ops else {}
            self.metadata[task]={'rollout_sources':result[1],
                'rollout_reader_fingerprint':fingerprint(result[2])}
            self.identities[task]=identity
        # ONLY explicit calibration capture may revisit fitted queries. Merely
        # enabling profiling or fixed plans does not waive evaluation leakage.
        return (*result[:3],set()) if self.diagnostic and self.trace_dir else result

    def install(self,runner):
        super().install(runner)
        old_variant=runner.runtime_variant
        runner.runtime_variant=lambda **kw: old_variant(**kw)+f'+v21-{self.layout}-{self.schedule}-packed{int(self.packed_ops)}-meta{int(self.metadata_cache)}-shadow{int(self.shadow_load)}'
        previous=runner.greedy_flexgen_completion
        runner.flexgen_sparse_decoder_logits=decoder_logits
        def completion(*args,**kw):
            model,loader=kw['model'],kw['loader']; period=kw['impress_selection_period_size']
            g,sources,readers,_=self._prepare(model,loader,period)
            task,digest=loader._info.task,query_hash(kw['query_token_ids'])
            header={'task':task,'query_hash':digest,'token_hash':loader._info.token_hash,
                'prefix_tokens':loader.prefix_tokens,'geometry':g,'period':period,
                'keep_ratio':float(loader.keep_ratio),'model_identity':loader._selector_index.payload['model_identity'],
                **self.metadata[task]}
            replay=None
            if self.replay_dir:
                if task not in self.replays:self.replays[task]=load_traces(self.replay_dir,task)
                replay=self.replays[task].get(digest)
                if replay is None:raise ValueError('Missing replay query; no online fallback')
                for f in ('token_hash','geometry','prefix_tokens','model_identity','period','keep_ratio'):
                    if replay[f]!=header[f]:raise ValueError(f'Replay identity mismatch: {f}')
            sink=TraceSink(self.trace_dir,task,digest,header,
                layers=range(g['layers']) if self.trace_layers is None else self.trace_layers,
                max_samples=self.max_samples) if self.trace_dir else None
            cfg={'packed':self.packed[task],'packed_ops':self.packed_ops,'layout':self.layout,
                'nvtx':self.annotate,'sink':sink,'replay':replay,'schedule':self.schedule,
                'shadow_load':self.shadow_load,'shadow_buffers':{},'selector_calls':0}
            pool=LengthPool() if self.metadata_cache else None
            tok=_SETTINGS.set(cfg); mtok=_ACTIVE_POOL.set(pool)
            try:
                result=previous(*args,**kw)
                if sink is not None:
                    sink.header['selection_by_layer']=[list(x) for x in loader._selected_by_layer];sink.save()
                if self.stats_path:
                    tensors=[x for r in readers.values() for x in (r.A,r.B,r.bk,r.bv)]
                    packed=[x for r in self.packed[task].values() for x in (r.A_transposed,r.B,r.bk_float,r.bv)]
                    shadow_bytes=sum(x.numel()*x.element_size() for kv in cfg['shadow_buffers'].values() for x in kv)
                    row={'task':task,'query_hash':digest,'layout':self.layout,'schedule':self.schedule,
                        'selector_calls_prefill':cfg['selector_calls'],'metadata_cache':self.metadata_cache,
                        'packed_reader_extra_storage_bytes':storage_bytes(tensors+packed)-storage_bytes(tensors),
                        'shadow_resolve_calls':len(cfg['shadow_buffers']),'shadow_buffer_bytes':shadow_bytes,
                        'fixed_plan_selector_compute_preserved':replay is not None or self.selection_plan_in_use,
                        'diagnostic':self.diagnostic,'phase_scope':'selector counts: first prefill; metadata: whole request incl label scoring',
                        **(pool.stats() if pool else {})}
                    extra=Path(str(self.stats_path)+'.v21.jsonl');extra.parent.mkdir(parents=True,exist_ok=True)
                    with extra.open('a') as f:f.write(json.dumps(row)+'\n')
                return result
            finally:
                _ACTIVE_POOL.reset(mtok);_SETTINGS.reset(tok)
        runner.greedy_flexgen_completion=completion
