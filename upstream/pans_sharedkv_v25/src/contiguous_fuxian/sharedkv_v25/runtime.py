"""Additive reprefill integration on the published v2.4 interfaces.

Only native repair chunks are requested from each consumer. Its logical token
selection and exact budget never change. Score/cache admission touches repair
chunks only; promoting the full consumer selection would destroy the saving.
"""
from __future__ import annotations
from contextvars import ContextVar
from pathlib import Path
import json,time
import torch
from contiguous_fuxian.sharedkv import runtime as v1
from contiguous_fuxian.sharedkv_v21 import runtime as v21
from contiguous_fuxian.sharedkv_v21.ops import pack,nvtx
from contiguous_fuxian.sharedkv.core import Reader,apply_rope
from contiguous_fuxian.sharedkv.io import load_safe,geometry,validate_sources,FORMAT as LEGACY_FORMAT
from contiguous_fuxian.sharedkv_v21.traces import fingerprint
from .policy import NativePolicy,partition,payload_account,FORMAT,BASE
from .attention import repaired_attention

_POLICY=ContextVar('v25_native_policy',default=None)
_EXTRA=ContextVar('v25_request_extra',default=None)

class RepairManager:
    def __init__(self,state,loader,policy):
        self.state,self.loader,self.policy=state,loader,policy
        self.parts={};self.pending={};self.buffers={};self.resolve_calls=0

    def configure(self,start,end,schedule):
        l=self.loader
        for lid in range(start,end):
            source=self.state.sources[lid]
            if l._selected_by_layer[lid]!=l._selected_by_layer[source]:
                raise ValueError('Source and consumer logical IDs differ; no union or token drop')
            if lid!=source:
                self.parts[lid]=partition(l._selected_by_layer[lid],l.prefix_tokens,self.policy)
        if schedule=='window':
            self.state.schedule_window(l,start,end)
            for lid in range(start,end):self.schedule(lid)
        elif schedule!='demand':raise ValueError('Unsupported scheduling policy')

    def schedule(self,lid):
        p=self.parts.get(lid)
        if p is None or not p.native_ids or lid in self.pending or lid in self.buffers:return
        # Published submission helper, but an independent pending dictionary.
        # prefetch_kind=current preserves native scheduler accounting.
        self.loader._submit_positions(lid,list(p.native_ids),self.pending,prefetch_kind='current')

    def resolve(self,lid,dtype,device):
        p=self.parts.get(lid)
        if p is None:return None,None,None
        if not p.native_ids:return None,None,p.cut
        if lid not in self.buffers:
            self.schedule(lid)
            k,v,ids=self.loader._resolve_entry(lid,self.pending.pop(lid))
            if ids.tolist()!=list(p.native_ids) or k.shape[0]!=len(p.native_ids) or k.shape!=v.shape:
                raise RuntimeError('Repair prefetch returned different physical positions; no hidden gather')
            k=k.to(device=device,dtype=dtype).contiguous()
            v=v.to(device=device,dtype=dtype).contiguous()
            self.buffers[lid]=(k.unsqueeze(0),v.unsqueeze(0));self.resolve_calls+=1
            # Logical selector metadata stays untouched. Only physically native
            # repair positions are admitted/scored in the consumer cache.
            updater=getattr(self.loader,'_cache_score_updater',None)
            if updater is not None:
                if getattr(self.loader,'defer_cache_score_updates',False):
                    raise ValueError('Deferred score updates are outside v25 contract')
                t=time.perf_counter();updates=updater(lid,list(p.native_ids),None)
                self.loader._cache_update_ms+=(time.perf_counter()-t)*1000
                self.loader._cache_score_updates+=int(updates)
                self.loader._cache_score_updated_layers.add(lid)
        k,v=self.buffers[lid]
        return k,v,p.cut

    def drain(self):
        # Do not leave asynchronous reads referencing a released request on error.
        for entry in list(self.pending.values()):entry[0].result()
        self.pending.clear()


def decoder_logits(*,model,input_ids,cache,position_start,loader,period_size,subperiod_size,
                   impress_selection_block_size=1,impress_period_prefetch_size=1,
                   impress_selection_period_size=1,impress_known_period_prefetch=False):
    state,cfg=v1._ACTIVE.get(),v21._SETTINGS.get();extra=_EXTRA.get();policy=_POLICY.get()
    if state is None or cfg is None or extra is None or policy is None:
        raise RuntimeError('Missing v25 request scope')
    if cfg.get('sink') is not None or cfg.get('replay') is not None or cfg.get('shadow_load'):
        raise ValueError('Use v24 teacher capture; v21 hidden-state traces/shadow are not v25-aware')
    cfg['state']=state
    if 'manager' not in extra:extra['manager']=RepairManager(state,loader,policy)
    manager=extra['manager'];core=model.model;hidden=core.embed_tokens(input_ids);b,t,_=hidden.shape
    if b!=1:raise ValueError('Only batch one supported')
    positions=torch.arange(position_start,position_start+t,device=hidden.device)
    cos,sin=core.rotary_emb(hidden,positions.unsqueeze(0));first=cache.get_seq_length(0)==0
    from contiguous_fuxian.flexgen_qwen_reprefill import configure_online_layer_selection
    g=state.geometry
    for lid,layer in enumerate(core.layers):
        if first and lid%impress_selection_period_size==0:
            end=min(g['layers'],lid+impress_selection_period_size)
            with nvtx(f'sharedkv/selector_{lid:02d}',cfg['nvtx']):
                period=configure_online_layer_selection(decoder_layer=layer,hidden_states=hidden,
                    position_embeddings=(cos,sin),layer_index=lid,loader=loader,period_size=period_size,
                    impress_selection_block_size=impress_selection_block_size,
                    impress_selection_period_size=impress_selection_period_size)
                if period!=impress_selection_period_size:raise ValueError('Adaptive period unsupported')
                cfg['selector_calls']+=1
            manager.configure(lid,end,cfg['schedule']);loader.prefetch_selector_keys(end)
        with nvtx(f'sharedkv/resolve_{lid:02d}',cfg['nvtx']):
            pk,pv=state.resolve(loader,lid,dtype=hidden.dtype,device=hidden.device)
            nk,nv,cut=manager.resolve(lid,hidden.dtype,hidden.device)
        residual=hidden;x=layer.input_layernorm(hidden);attn=layer.self_attn
        q=attn.q_proj(x).view(b,t,g['q_heads'],g['head_dim'])
        k=attn.k_proj(x).view(b,t,g['kv_heads'],g['head_dim'])
        v=attn.v_proj(x).view(b,t,g['kv_heads'],g['head_dim'])
        q,k=apply_rope(q,cos,sin),apply_rope(k,cos,sin)
        tk,tv=cache.update(k.transpose(1,2),v.transpose(1,2),lid,
                           {'cos':cos,'sin':sin,'cache_position':positions})
        state.tail_lengths[lid]=max(state.tail_lengths.get(lid,0),int(tk.shape[2]))
        with nvtx(f'sharedkv/attention_{lid:02d}',cfg['nvtx']):
            out=repaired_attention(q,pk,pv,nk,nv,tk.transpose(1,2),tv.transpose(1,2),
                cut=pk.shape[1] if cut is None else cut,reader=state.readers.get(lid),
                packed=cfg['packed'].get(lid) if cfg['packed_ops'] else None,
                backend=state.backend,layout=cfg['layout'],annotate=cfg['nvtx'])
        hidden=residual+attn.o_proj(out.reshape(b,t,-1).contiguous())
        hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
    return model.lm_head(core.norm(hidden))


class Runtime(v21.Runtime):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.policies={}

    def _prepare(self,model,loader,period):
        task=loader._info.task
        identity=(id(model),loader._info.token_hash,loader.prefix_tokens,float(loader.keep_ratio),period)
        if task in self.prepared:
            if self.identities[task]!=identity:raise ValueError('Runtime configuration changed')
            return self.prepared[task]
        if self.mode=='identity':
            result=super()._prepare(model,loader,period)
            self.policies[task]=NativePolicy(0,loader.chunk_size)
            return result
        bundle=load_safe(self.adapter_dir/f'{task}.pt')
        if bundle.get('format')==LEGACY_FORMAT:
            result=super()._prepare(model,loader,period)
            self.policies[task]=NativePolicy(0,loader.chunk_size)
            return result
        if bundle.get('format')!=FORMAT:raise ValueError('Unknown repaired-prefix bundle format')
        if bundle.get('audited_runtime_base')!=BASE:
            raise ValueError('Repaired-prefix checkpoint was exported for a different audited runtime')
        if not (self.diagnostic or self.allow_candidate):
            raise ValueError('v25 is research-only; explicit --allow-candidate required')
        policy=NativePolicy(**bundle['native_policy'])
        if policy.chunk_size!=loader.chunk_size:raise ValueError('Repair chunk size mismatch')
        g=geometry(model.config)
        if model.config.model_type!='qwen2' or getattr(model.config,'use_sliding_window',False) or getattr(model.config,'rope_scaling',None) is not None:
            raise ValueError('Only fixed-RoPE full-attention Qwen2 supported')
        for layer in model.model.layers:
            if getattr(layer.self_attn,'sliding_window',None) is not None or hasattr(layer.self_attn,'q_norm') or hasattr(layer.self_attn,'k_norm'):
                raise ValueError('Unexpected attention architecture')
        device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
        if dtype!=torch.bfloat16:raise ValueError('Published online path requires BF16 compute')
        expected={'task':task,'token_hash':loader._info.token_hash,'prefix_tokens':loader.prefix_tokens,
                  'geometry':g,'period':period,'keep_ratio':float(loader.keep_ratio)}
        for k,v in expected.items():
            if bundle[k]!=v:raise ValueError(f'Reader identity mismatch: {k}')
        if loader._selector_index is None or bundle['model_identity']!=loader._selector_index.payload['model_identity']:
            raise ValueError('Model/adapter/config fingerprint mismatch')
        sources=list(bundle['sources']);validate_sources(sources,period=period)
        if len(sources)!=g['layers']:raise ValueError('Source geometry mismatch')
        readers={int(l):Reader.from_state_dict(r).to(device=device,dtype=dtype) for l,r in bundle['readers'].items()}
        if set(readers)!={l for l,s in enumerate(sources) if l!=s}:raise ValueError('Missing or extra reader')
        for r in readers.values():
            if tuple(r.A.shape)!=(g['kv_heads'],g['head_dim'],g['head_dim']):raise ValueError('Reader head geometry mismatch')
        if self.backend in ('flash2','vllm2'):
            from contiguous_fuxian.sharedkv_v21.backend import attention
            q=torch.zeros(1,2,g['q_heads'],g['head_dim'],device=device,dtype=dtype)
            k=torch.zeros(1,3,g['kv_heads'],g['head_dim'],device=device,dtype=dtype)
            attention(q,k,k,causal=True,backend=self.backend);torch.cuda.synchronize()
        result=(g,sources,readers,set(bundle['calibration_query_hashes']))
        self.prepared[task]=result;self.identities[task]=identity;self.policies[task]=policy
        self.packed[task]={lid:pack(r) for lid,r in readers.items()} if self.packed_ops else {}
        self.metadata[task]={'rollout_sources':sources,'rollout_reader_fingerprint':fingerprint(readers)}
        return result

    def install(self,runner):
        super().install(runner)
        previous=runner.greedy_flexgen_completion
        runner.flexgen_sparse_decoder_logits=decoder_logits
        old_variant=runner.runtime_variant
        runner.runtime_variant=lambda **kw:old_variant(**kw)+'+v25-native-selected-suffix'
        def completion(*args,**kw):
            loader=kw['loader'];model=kw['model'];period=kw['impress_selection_period_size']
            if getattr(loader,'defer_cache_score_updates',False):raise ValueError('Deferred score updates unsupported')
            self._prepare(model,loader,period);policy=self.policies[loader._info.task]
            extra={};pt=_POLICY.set(policy);et=_EXTRA.set(extra)
            try:
                result=previous(*args,**kw)
                manager=extra.get('manager')
                if self.stats_path and manager is not None:
                    state=manager.state;g=state.geometry
                    row=payload_account(loader._selected_by_layer,state.sources,loader.prefix_tokens,
                                        g['kv_heads'],g['head_dim'],policy)
                    row.update(task=loader._info.task,query_hash=v1.query_hash(kw['query_token_ids']),
                        native_policy=policy.state(),native_repair_resolve_calls=manager.resolve_calls,
                        full_source_resolve_calls=state.resolve_calls,
                        total_resolve_calls=manager.resolve_calls+state.resolve_calls,
                        native_repair_buffer_bytes=sum(x.numel()*x.element_size() for kv in manager.buffers.values() for x in kv),
                        cuts_by_consumer={str(l):p.cut for l,p in manager.parts.items()},
                        native_repair_storage_ptrs={str(l):[k.data_ptr(),v.data_ptr()] for l,(k,v) in manager.buffers.items()},
                        shared_base_storage_ptrs={str(l):[state.bank[src][0].data_ptr(),state.bank[src][1].data_ptr()] for l,src in enumerate(state.sources)},
                        full_source_buffer_bytes=sum(x.numel()*x.element_size() for kv in state.bank.values() for x in kv),
                        diagnostic=self.diagnostic,quality_certified=False,
                        legacy_stats_omit_native_repair_buffers=True)
                    path=Path(str(self.stats_path)+'.v25.jsonl')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('a') as f:f.write(json.dumps(row)+'\n')
                return result
            finally:
                try:
                    if 'manager' in extra:extra['manager'].drain()
                finally:_EXTRA.reset(et);_POLICY.reset(pt)
        runner.greedy_flexgen_completion=completion
