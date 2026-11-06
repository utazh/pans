"""Additive v2 runtime extension. Existing P8 selection and source ownership are unchanged.

A new on-disk format prevents legacy launchers silently ignoring head maps.
A private, temporary compatibility view is used solely for v2 parameter loading;
it is never an output artifact and never changes the user's checkpoint.
"""
from __future__ import annotations
from contextvars import ContextVar
from pathlib import Path
import copy
import json
import tempfile
import torch
from contiguous_fuxian.sharedkv_v21 import runtime as parent
from contiguous_fuxian.sharedkv.io import load_safe, FORMAT as OLD_FORMAT, query_hash
from .matching import FORMAT, normalized_maps, aligned_fingerprint
from .ops import prepare_route,mapped_attention

_ROUTES=ContextVar('sharedkv_v22_routes',default=None)
_ORIGINAL_DISPATCH=parent.dispatch


def route_dispatch(q,pk,pv,tk,tv,*,reader=None,packed=None,backend='reference',layout='split',annotate=False):
    routes=_ROUTES.get()
    route=None if routes is None or reader is None else routes.get(id(reader))
    if route is None or route.identity:
        return _ORIGINAL_DISPATCH(q,pk,pv,tk,tv,reader=reader,packed=packed,
                                  backend=backend,layout=layout,annotate=annotate)
    return mapped_attention(q,pk,pv,tk,tv,reader=reader,route=route,packed=packed,
                            backend=backend,annotate=annotate)


class Runtime(parent.Runtime):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.head_maps={};self.routes={};self._view=tempfile.TemporaryDirectory(prefix='pans_v22_')

    def _prepare(self,model,loader,period):
        task=loader._info.task
        if task not in self.routes:
            raw=None
            if self.mode=='shared':
                raw=load_safe(self.adapter_dir/f'{task}.pt')
                if raw.get('format') not in (FORMAT,OLD_FORMAT):raise ValueError('Unknown reader format')
                if raw.get('format')!=FORMAT and raw.get('head_maps'):
                    raise ValueError('A head-mapped bundle must use the v22 format')
                consumers=[i for i,s in enumerate(raw['sources']) if i!=s]
                maps=normalized_maps(raw.get('head_maps',{}),consumers,raw['geometry']['kv_heads'])
            else:
                maps={}
            original_dir=self.adapter_dir
            try:
                if raw is not None and raw['format']==FORMAT:
                    converted=copy.copy(raw);converted['format']=OLD_FORMAT
                    view=Path(self._view.name);view.mkdir(exist_ok=True)
                    torch.save(converted,view/f'{task}.pt')
                    self.adapter_dir=view
                result=super()._prepare(model,loader,period)
            finally:
                self.adapter_dir=original_dir
            g,sources,readers,banned=result
            device=next(model.parameters()).device
            self.head_maps[task]=maps
            self.routes[task]={id(readers[lid]):prepare_route(p,q_heads=g['q_heads'],device=device)
                                for lid,p in maps.items()}
            # Propagates into paired traces via v21's **self.metadata[task] header.
            self.metadata[task]['rollout_head_maps']={str(l):list(p) for l,p in maps.items()}
            self.metadata[task]['rollout_reader_fingerprint']=aligned_fingerprint(readers,maps)
            return result
        return super()._prepare(model,loader,period)

    def install(self,runner):
        # Single-threaded experiment launcher only, as in v2. ContextVar prevents
        # one request's head routing from affecting another runtime's dispatch.
        if parent.dispatch not in (_ORIGINAL_DISPATCH,route_dispatch):
            raise RuntimeError('An unexpected dispatch patch is already installed')
        parent.dispatch=route_dispatch
        super().install(runner)
        previous=runner.greedy_flexgen_completion
        old_variant=runner.runtime_variant
        runner.runtime_variant=lambda **kw:old_variant(**kw)+'+v22-head-aligned'
        def completion(*args,**kw):
            self._prepare(kw['model'],kw['loader'],kw['impress_selection_period_size'])
            task=kw['loader']._info.task
            token=_ROUTES.set(self.routes[task])
            try:
                result=previous(*args,**kw)
                if self.stats_path:
                    row={'task':task,'query_hash':query_hash(kw['query_token_ids']),
                         'head_maps':{str(l):list(p) for l,p in self.head_maps[task].items()},
                         'resident_head_route_bytes':sum(r.resident_bytes for r in self.routes[task].values()),
                         'nonidentity_route_consumers':sum(not r.identity for r in self.routes[task].values()),
                         'prefix_kv_reordered_online':False,
                         'note':'Additional temporary Q/output/LSE gathers occur inside TTFT; no hardware byte inference from this row.'}
                    path=Path(str(self.stats_path)+'.v22.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('a') as f:f.write(json.dumps(row)+'\n')
                return result
            finally:_ROUTES.reset(token)
        runner.greedy_flexgen_completion=completion
