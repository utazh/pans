"""Route SHORT Q/output tensors, never gather/reorder/expand source prefix K/V.

All public attention tensors are [B,T,Hq,D] or [B,N,Hkv,D].
p[target_kv_head] = source_kv_head. K and V use the SAME bijection.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .matching import validate_permutation
from contiguous_fuxian.sharedkv.core import Reader, query_transform, value_transform, key_offset, merge_segments
from contiguous_fuxian.sharedkv_v21 import ops as old_ops
from contiguous_fuxian.sharedkv_v21.backend import attention as default_attention


@dataclass(frozen=True)
class HeadRoute:
    permutation: tuple[int, ...]
    to_source: torch.Tensor
    to_target: torch.Tensor
    identity: bool

    @property
    def resident_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.to_source, self.to_target))


def prepare_route(p, *, q_heads: int, device) -> HeadRoute:
    p = validate_permutation(p, len(p))
    if q_heads <= 0 or q_heads % len(p):
        raise ValueError('Q heads must divide into complete physical GQA groups')
    group = q_heads // len(p)
    inverse = [p.index(s) for s in range(len(p))]
    # Arrange complete Q groups in SOURCE order before the one prefix attention.
    to_source = torch.tensor([h*group+j for h in inverse for j in range(group)],
                             dtype=torch.long, device=device)
    # Bring its output/LSE back to TARGET order before B, tail merge and o_proj.
    to_target = torch.tensor([s*group+j for s in p for j in range(group)],
                             dtype=torch.long, device=device)
    return HeadRoute(p, to_source, to_target, p == tuple(range(len(p))))


def mapped_attention(q, pk, pv, tk, tv, *, reader: Reader, route: HeadRoute,
                     packed=None, backend='reference', annotate=False, attention_fn=None):
    fn = default_attention if attention_fn is None else attention_fn
    if q.shape[2] != route.to_target.numel() or pk.shape[2] != len(route.permutation):
        raise ValueError('Route/attention geometry mismatch')
    if route.identity and attention_fn is None:
        return old_ops.dispatch(q,pk,pv,tk,tv,reader=reader,packed=packed,
                                backend=backend,layout='split',annotate=annotate)
    with old_ops.nvtx('headalign/q_transform_route', annotate):
        if packed is None:
            qr = query_transform(q,reader)
            offset = key_offset(q,reader,q.shape[-1]**-0.5)
        else:
            qr,offset = old_ops.prepare_query(q,packed.A_transposed,packed.bk_float)
        if not route.identity:
            qr = qr.index_select(2,route.to_source)
    with old_ops.nvtx('headalign/shared_prefix_attention', annotate):
        # Intentionally pass these EXACT objects; no per-consumer KV permutation.
        up,lp = fn(qr,pk,pv,backend=backend)
    with old_ops.nvtx('headalign/output_route', annotate):
        if not route.identity:
            up = up.index_select(2,route.to_target)
            lp = lp.index_select(1,route.to_target)
    with old_ops.nvtx('headalign/private_attention', annotate):
        ut,lt = fn(q,tk,tv,backend=backend,causal=True)
    with old_ops.nvtx('headalign/value_merge', annotate):
        if packed is None:
            return merge_segments(value_transform(up,reader),lp+offset.to(lp.dtype),ut,lt)[0]
        return old_ops.finish(up,lp,ut,lt,offset,packed.B,packed.bv)
