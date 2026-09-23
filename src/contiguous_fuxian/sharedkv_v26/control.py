"""Fixed-reader native-patch intervention; diagnostic only, not a speed method.

The source-proxy arm still loads the original native patch, but replaces its
contents at attention time by the affine source prediction at the SAME token
positions. It reconstructs only the small diagnostic patch (bounded explicitly),
never a whole consumer prefix. This is not a retrained R0 baseline.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch

@dataclass
class ControlStats:
    consumer_calls: int = 0
    replaced_patch_tokens: int = 0
    max_patch_tokens: int = 0
    proxy_elements_materialized: int = 0


def source_proxy(source_k: torch.Tensor, source_v: torch.Tensor, cut: int,
                 reader: Any, *, max_tokens: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    if source_k.ndim != 4 or source_k.shape != source_v.shape:
        raise ValueError('Expected [B,N,Hkv,D] source K/V')
    if type(cut) is not int or not 0 < cut < source_k.shape[1]:
        raise ValueError('Proxy requires a nonempty SHARED prefix and a bounded nonempty patch')
    count = source_k.shape[1] - cut
    if max_tokens <= 0 or count > max_tokens:
        raise ValueError('Diagnostic patch exceeds max_tokens; full-prefix reconstruction is forbidden')
    h, d = source_k.shape[2:]
    if reader is None or tuple(reader.A.shape) != (h, d, d) or tuple(reader.B.shape) != (h, d, d):
        raise ValueError('Reader geometry differs')
    if source_k.device != source_v.device or source_k.dtype != source_v.dtype:
        raise ValueError('Mixed source geometry')
    # These allocations are intentional and confined to the SMALL intervention.
    # FP64 is used by identity tests. BF16 rounding differs from absorbed R0.
    a, b = reader.A.to(source_k), reader.B.to(source_v)
    bk, bv = reader.bk.to(source_k), reader.bv.to(source_v)
    pk = torch.einsum('bnhd,hde->bnhe', source_k[:, cut:], a) + bk[None, None]
    pv = torch.einsum('bnhd,hde->bnhe', source_v[:, cut:], b) + bv[None, None]
    return pk.contiguous(), pv.contiguous()


def controlled_attention(original, q, source_k, source_v, native_k, native_v,
                         tail_k, tail_v, *, repair_values='native', stats=None,
                         max_proxy_tokens=128, **kwargs):
    if repair_values not in ('native', 'source-proxy'):
        raise ValueError('Unknown repair intervention')
    if repair_values == 'source-proxy' and kwargs.get('reader') is not None:
        if native_k is None or native_v is None:
            raise ValueError('source-proxy requires an R>0 checkpoint; do not reinterpret R0')
        cut = kwargs['cut']
        if source_k.shape[1] - cut != native_k.shape[1]:
            raise ValueError('The proxy and original patch must have identical positions/counts')
        native_k, native_v = source_proxy(source_k, source_v, cut, kwargs['reader'],
                                         max_tokens=max_proxy_tokens)
        if stats is not None:
            stats.consumer_calls += 1
            stats.replaced_patch_tokens += native_k.shape[1]
            stats.max_patch_tokens = max(stats.max_patch_tokens, native_k.shape[1])
            stats.proxy_elements_materialized += native_k.numel() + native_v.numel()
    return original(q, source_k, source_v, native_k, native_v, tail_k, tail_v, **kwargs)
