"""The shared source is sliced by VIEW; native repair joins the private segment.

There are still two attention calls per consumer, not three. The longer private
segment causes additional concat/compute and remains inside latency timing.
"""
from __future__ import annotations
import torch


def repaired_attention(q, source_k, source_v, native_k, native_v, tail_k, tail_v, *,
                       cut: int, reader=None, packed=None, backend='reference',
                       layout='hybrid', annotate=False):
    from contiguous_fuxian.sharedkv_v21.ops import dispatch
    if reader is None:
        if native_k is not None or native_v is not None:
            raise ValueError('Native owner does not need a consumer repair buffer')
        return dispatch(q,source_k,source_v,tail_k,tail_v,backend=backend,layout=layout,annotate=annotate)
    if not 0 <= cut <= source_k.shape[1]:
        raise ValueError('Invalid shared view length')
    if native_k is None:
        if cut != source_k.shape[1] or native_v is not None:
            raise ValueError('Missing repair buffer')
        return dispatch(q,source_k,source_v,tail_k,tail_v,reader=reader,packed=packed,
                        backend=backend,layout=layout,annotate=annotate)
    if native_v is None or native_k.shape != native_v.shape:
        raise ValueError('Invalid repair K/V')
    if native_k.shape[1] != source_k.shape[1]-cut:
        raise ValueError('Duplicate/missing original prefix positions')
    ek, ev = torch.cat((native_k,tail_k),1), torch.cat((native_v,tail_v),1)
    if cut == 0:
        # Mathematical endpoint test only; policy forbids a full native fallback.
        from contiguous_fuxian.sharedkv_v21.backend import attention
        return attention(q,ek,ev,causal=True,backend=backend)[0]
    return dispatch(q,source_k[:,:cut],source_v[:,:cut],ek,ev,reader=reader,packed=packed,
                    backend=backend,layout=layout,annotate=annotate)
