"""Equivalent reader algebra and optional hybrid attention.

No tensor-valued validation or CUDA synchronization in the hot path.
Hybrid concatenates ONLY unshared/native layers; shared consumers still read
an immutable source buffer without reconstructing their layer-specific KV.
Hybrid may be slower when concatenation costs dominate: benchmark both modes.
"""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import nullcontext
import torch
from contiguous_fuxian.sharedkv.core import Reader, query_transform, value_transform, key_offset, merge_segments
from .backend import attention as attention_with_lse


@dataclass(frozen=True)
class PackedReader:
    A_transposed: torch.Tensor
    B: torch.Tensor
    bk_float: torch.Tensor
    bv: torch.Tensor


def pack(reader: Reader) -> PackedReader:
    # Call at model/adapter preparation, NOT per attention call.
    reader.validate()
    dtype = torch.float64 if reader.A.dtype == torch.float64 else torch.float32
    return PackedReader(reader.A.transpose(-1, -2).contiguous(), reader.B.contiguous(),
                        reader.bk.to(dtype).contiguous(), reader.bv.contiguous())


def prepare_query(q, at, bk):
    b,t,hq,d = q.shape
    hk = at.shape[0]
    # [B, KV-head, T, GQA-group, D]; no head-repeat of the source KV.
    x = q.reshape(b,t,hk,hq//hk,d).permute(0,2,1,3,4).reshape(b,hk,t*(hq//hk),d)
    qr = torch.matmul(x, at).reshape(b,hk,t,hq//hk,d).permute(0,2,1,3,4).reshape_as(q).contiguous()
    offset = (x.to(bk.dtype)*bk[None,:,None,:]).sum(-1)
    offset = offset.reshape(b,hk,t,hq//hk).permute(0,1,3,2).reshape(b,hq,t)*(d**-0.5)
    return qr, offset


def finish(up, lp, ut, lt, offset, B, bv):
    b,t,hq,d = up.shape
    hk = B.shape[0]
    x = up.reshape(b,t,hk,hq//hk,d).permute(0,2,1,3,4).reshape(b,hk,t*(hq//hk),d)
    y = torch.matmul(x, B).reshape(b,hk,t,hq//hk,d).permute(0,2,1,3,4)
    # Keep the v1 bias rounding order; mixed output accumulated in FP32.
    y = (y+bv[None,None,:,None,:]).reshape_as(up)
    z = torch.logaddexp(lp+offset, lt)
    beta = (lp+offset-z).exp().transpose(1,2).unsqueeze(-1)
    dtype = torch.float64 if up.dtype == torch.float64 else torch.float32
    return (beta.to(dtype)*y.to(dtype)+(1-beta.to(dtype))*ut.to(dtype)).to(up.dtype)


def nvtx(name, enabled=False):
    return torch.cuda.nvtx.range(name) if enabled else nullcontext()


def packed_attention(q, pk, pv, tk, tv, reader: PackedReader, *, backend="reference", annotate=False):
    with nvtx("sharedkv/reader_q_offset", annotate):
        qr, offset = prepare_query(q, reader.A_transposed, reader.bk_float)
    with nvtx("sharedkv/prefix_attention", annotate):
        up, lp = attention_with_lse(qr,pk,pv,backend=backend)
    with nvtx("sharedkv/private_attention", annotate):
        ut, lt = attention_with_lse(q,tk,tv,causal=True,backend=backend)
    with nvtx("sharedkv/reader_v_merge", annotate):
        return finish(up,lp,ut,lt,offset,reader.B,reader.bv)


def native_attention(q, pk, pv, tk, tv, *, backend="reference", annotate=False):
    # Bottom-right causal masking: all prefix visible + causal private tail.
    # A temporary GPU concatenation is deliberately INCLUDED in this path.
    with nvtx("sharedkv/native_cat", annotate):
        k, v = torch.cat((pk,tk),1), torch.cat((pv,tv),1)
    with nvtx("sharedkv/native_attention", annotate):
        return attention_with_lse(q,k,v,causal=True,backend=backend)[0]


def dispatch(q,pk,pv,tk,tv, *, reader=None, packed=None, backend="reference", layout="split", annotate=False):
    if layout not in ("split","hybrid"):
        raise ValueError("layout must be split or hybrid")
    if reader is None:
        if layout == "hybrid":
            return native_attention(q,pk,pv,tk,tv,backend=backend,annotate=annotate)
        return segmented_attention(q,pk,pv,tk,tv,backend=backend)
    if packed is None:
        return segmented_attention(q,pk,pv,tk,tv,reader=reader,backend=backend)
    return packed_attention(q,pk,pv,tk,tv,packed,backend=backend,annotate=annotate)


def segmented_attention(q,pk,pv,tk,tv,*,reader=None,backend="reference"):
    qr=q if reader is None else query_transform(q,reader)
    up,lp=attention_with_lse(qr,pk,pv,backend=backend)
    if reader is not None:
        up=value_transform(up,reader)
        lp=lp+key_offset(q,reader,q.shape[-1]**-.5).to(lp.dtype)
    ut,lt=attention_with_lse(q,tk,tv,causal=True,backend=backend)
    return merge_segments(up,lp,ut,lt)[0]
