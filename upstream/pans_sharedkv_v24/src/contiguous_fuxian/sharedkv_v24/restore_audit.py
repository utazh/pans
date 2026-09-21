"""Diagnostic read-path attribution, NOT a faster admission implementation.

Records disk-read attempts at native prefetch/promotion entry points. Byte
counts are declared file sizes, NOT block-device/NAND/H2D counters. No cache
policy, cold-state test, score update, or I/O advice is removed.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import json,math,threading,time
import numpy as np
import torch
from .records import digest_ids

_ACTIVE=None
_LOCK=threading.Lock()

class ReadObserver:
    def __init__(self,module):
        self.module=module;self.counts=Counter();self.sizes={};self.lock=threading.Lock()
    def note(self,role,token):
        path=str(token.path);size=math.prod(token.shape)*np.dtype(token.dtype).itemsize
        with self.lock:
            self.counts[(role,path)]+=1;self.sizes[path]=size
    def report(self):
        roles={role:{p for r,p in self.counts if r==role} for role in ('prefetch','promotion')}
        out={}
        for role,paths in roles.items():
            out[role]={'calls':sum(self.counts[(role,p)] for p in paths),'unique_files':len(paths),
                'declared_file_bytes_requested':sum(self.counts[(role,p)]*self.sizes[p] for p in paths)}
        overlap=roles['prefetch']&roles['promotion']
        out.update(files_seen_in_both_paths=len(overlap),
            declared_overlap_file_bytes=sum(self.sizes[p] for p in overlap),
            repeated_path_examples=[{'path':p,'prefetch_calls':self.counts[('prefetch',p)],
                                     'promotion_calls':self.counts[('promotion',p)]}
                                    for p in sorted(overlap)[:32]],
            formal_timing_eligible=False,
            note='Read attempts and declared bytes; not proof each attempt reached disk. Correlate v23 proc I/O and Nsight.')
        return out
    def __enter__(self):
        global _ACTIVE
        with _LOCK:
            if _ACTIVE is not None:raise RuntimeError('Concurrent/nested diagnostic runs unsupported')
            _ACTIVE=self
        cls=self.module.PrefixKVLayer;token_cls=self.module.TokenCacheDisk
        self.old_prefetch=cls._chunk_source_readonly;self.old_to=token_cls.to
        oldp,oldt=self.old_prefetch,self.old_to
        observer=self
        def prefetch(layer,chunk,want_key):
            token=chunk.token
            if token.device=='disk':observer.note('prefetch',token)
            return oldp(layer,chunk,want_key)
        def move(token,device):
            if token.device=='disk' and str(device)!='disk':observer.note('promotion',token)
            return oldt(token,device)
        cls._chunk_source_readonly=prefetch;token_cls.to=move
        return self
    def __exit__(self,*exc):
        global _ACTIVE
        self.module.PrefixKVLayer._chunk_source_readonly=self.old_prefetch
        self.module.TokenCacheDisk.to=self.old_to
        with _LOCK:_ACTIVE=None


def install(runner,path):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    previous=runner.flexgen_sparse_decoder_logits
    def decoder(*args,**kw):
        if args:raise ValueError('Keyword-only decoder expected')
        if kw['cache'].get_seq_length(0)!=0:return previous(*args,**kw)
        import my_pcache_fast as module
        observer=ReadObserver(module);t0=time.perf_counter()
        with observer:
            result=previous(*args,**kw)
            pcache=kw['loader']._pcache
            executor=getattr(pcache,'_prefetch_executor',None)
            if executor is not None:
                # Diagnostic barrier only: drain existing queue before restoring wrappers.
                executor.submit(lambda:None,priority=10**12,label='v24_audit_fence').result()
            if kw['input_ids'].device.type=='cuda':torch.cuda.synchronize()
        row={'query_hash':digest_ids(kw['input_ids'][0].detach().cpu().tolist()),
             'task':kw['loader']._info.task,'diagnostic_decoder_wall_ms':(time.perf_counter()-t0)*1000,
             **observer.report()}
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:f.write(json.dumps(row)+'\n')
        return result
    runner.flexgen_sparse_decoder_logits=decoder
