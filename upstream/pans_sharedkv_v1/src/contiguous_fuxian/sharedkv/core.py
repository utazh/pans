"""Read shared physical KV without reconstructing per-layer prefix tensors.

All public attention tensors use [batch, token, head, feature]. Readers use
[physical_kv_head, input_feature, output_feature]. Runtime is inference-only.
The reference backend is for correctness, NOT a latency benchmark backend.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Literal
import torch

Backend = Literal["reference", "flash2"]


@dataclass(frozen=True)
class Reader:
    A: torch.Tensor
    B: torch.Tensor
    bk: torch.Tensor
    bv: torch.Tensor

    def validate(self) -> None:
        if self.A.ndim != 3 or self.A.shape[-1] != self.A.shape[-2]:
            raise ValueError("A must be [KV heads, d, d]")
        h, d, _ = self.A.shape
        if self.B.shape != (h, d, d) or self.bk.shape != (h, d) or self.bv.shape != (h, d):
            raise ValueError("Reader geometry mismatch")
        for t in (self.A, self.B, self.bk, self.bv):
            if not torch.is_floating_point(t) or not torch.isfinite(t).all():
                raise ValueError("Reader parameters must be finite floating tensors")

    def to(self, *, device, dtype) -> "Reader":
        return Reader(*(x.to(device=device, dtype=dtype).contiguous()
                        for x in (self.A, self.B, self.bk, self.bv)))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).detach().cpu() for name in ("A", "B", "bk", "bv")}

    @classmethod
    def from_state_dict(cls, x: dict) -> "Reader":
        result = cls(*(x[n] for n in ("A", "B", "bk", "bv")))
        result.validate()
        return result


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen2 split-half RoPE pairs j with j+d/2, NOT adjacent channels."""
    if x.shape[-1] % 2:
        raise ValueError("RoPE dimension must be even")
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B,T,H,D]; cos/sin: [B,T,D], supplied by the original model.
    return x * cos.unsqueeze(2) + rotate_half(x) * sin.unsqueeze(2)


def query_transform(q: torch.Tensor, reader: Reader) -> torch.Tensor:
    b, t, hq, d = q.shape
    hk = reader.A.shape[0]
    if hq % hk or reader.A.shape[-1] != d:
        raise ValueError("Reader does not match the GQA geometry")
    # K_hat = K_source @ A, so Q_read = Q @ A.T.
    return torch.einsum("bthgj,hij->bthgi", q.reshape(b, t, hk, hq // hk, d),
                        reader.A).reshape_as(q).contiguous()


def value_transform(u: torch.Tensor, reader: Reader) -> torch.Tensor:
    b, t, hq, d = u.shape
    hk = reader.B.shape[0]
    z = torch.einsum("bthgi,hij->bthgj", u.reshape(b, t, hk, hq // hk, d), reader.B)
    return (z + reader.bv[None, None, :, None, :]).reshape_as(u)


def key_offset(q: torch.Tensor, reader: Reader, scale: float) -> torch.Tensor:
    b, t, hq, d = q.shape
    hk = reader.bk.shape[0]
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    z = torch.einsum("bthgj,hj->bthg", q.reshape(b, t, hk, hq // hk, d).to(dtype), reader.bk.to(dtype))
    return z.reshape(b, t, hq).transpose(1, 2) * scale


def _geometry(q, k, v):
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("Expected q/k/v [B,T,H,D]; K and V must have equal shapes")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1] or q.shape[2] % k.shape[2]:
        raise ValueError("Batch/feature/GQA geometry mismatch")
    if not q.shape[1] or not k.shape[1]:
        raise ValueError("Empty attention segments are not supported")
    if not (q.device == k.device == v.device and q.dtype == k.dtype == v.dtype):
        raise ValueError("q/k/v must have identical device and dtype")


def attention_with_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
                       causal: bool = False, backend: Backend = "reference",
                       q_positions: torch.Tensor | None = None,
                       query_chunk: int = 16, key_chunk: int = 1024):
    """Return conditional segment output and natural-log Z [B,Hq,T].

    Causal alignment is bottom-right: q row i sees keys <= N-T+i.
    q_positions overrides these positions in reference mode (calibration).
    """
    _geometry(q, k, v)
    scale = q.shape[-1] ** -0.5
    if backend == "flash2":
        if q_positions is not None:
            raise ValueError("Sampled query positions are reference-only")
        if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("flash2 requires CUDA fp16/bf16")
        # Public API. At dropout_p=0, upstream gates return_softmax to false in
        # the CUDA kernel; output+LSE are returned without a quadratic mask.
        # Never silently fall back to reference when doing timing.
        from flash_attn import flash_attn_func
        out, lse, mask = flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(),
                                       dropout_p=0.0, softmax_scale=scale,
                                       causal=causal, return_attn_probs=True)
        if mask is not None and mask.numel() != 0:
            raise RuntimeError("Unexpected quadratic attention-mask output; unsupported FlashAttention build")
        if lse.shape != (q.shape[0], q.shape[2], q.shape[1]):
            raise RuntimeError("Unexpected FlashAttention LSE shape")
        return out, lse
    if backend != "reference":
        raise ValueError(f"Unknown backend: {backend}")
    if query_chunk <= 0 or key_chunk <= 0:
        raise ValueError("Chunk sizes must be positive")
    b, tq, hq, d = q.shape
    n, hk = k.shape[1:3]
    groups = hq // hk
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    outputs, lses = [], []
    if q_positions is None:
        q_positions = torch.arange(n - tq, n, device=q.device)
    if q_positions.shape != (tq,):
        raise ValueError("q_positions must be [query_tokens]")
    if causal and (q_positions.min() < 0 or q_positions.max() >= n):
        raise ValueError("Every causal query must have at least one visible key")
    for start in range(0, tq, query_chunk):
        end = min(start + query_chunk, tq)
        qc = q[:, start:end].to(dtype).reshape(b, end-start, hk, groups, d).permute(0, 2, 3, 1, 4)
        shape = (b, hk, groups, end-start)
        maximum = torch.full(shape, -torch.inf, device=q.device, dtype=dtype)
        denom = torch.zeros(shape, device=q.device, dtype=dtype)
        numer = torch.zeros((*shape, d), device=q.device, dtype=dtype)
        for lo in range(0, n, key_chunk):
            hi = min(lo + key_chunk, n)
            kc = k[:, lo:hi].to(dtype).permute(0, 2, 1, 3).unsqueeze(2)
            vc = v[:, lo:hi].to(dtype).permute(0, 2, 1, 3).unsqueeze(2)
            scores = torch.matmul(qc, kc.transpose(-1, -2)) * scale
            if causal:
                allowed = torch.arange(lo, hi, device=q.device)[None, :] <= q_positions[start:end, None]
                scores = scores.masked_fill(~allowed[None, None, None], -torch.inf)
            new_max = torch.maximum(maximum, scores.amax(dim=-1))
            alpha = torch.exp(maximum - new_max)
            p = torch.exp(scores - new_max.unsqueeze(-1))
            numer = numer * alpha.unsqueeze(-1) + torch.matmul(p, vc)
            denom = denom * alpha + p.sum(dim=-1)
            maximum = new_max
        out = (numer / denom.unsqueeze(-1)).permute(0, 3, 1, 2, 4).reshape(b, end-start, hq, d)
        lse = (maximum + denom.log()).reshape(b, hq, end-start)
        outputs.append(out.to(q.dtype))
        lses.append(lse)
    return torch.cat(outputs, dim=1), torch.cat(lses, dim=-1)


def merge_segments(up, lp, ut, lt):
    """Joint softmax, not a sum of separately-normalized attention outputs."""
    logz = torch.logaddexp(lp, lt)
    beta = (lp - logz).exp().transpose(1, 2).unsqueeze(-1)
    # Do accumulation in float32 (or float64 for mathematical unit tests).
    dtype = torch.float64 if up.dtype == torch.float64 else torch.float32
    result = beta.to(dtype) * up.to(dtype) + (1 - beta.to(dtype)) * ut.to(dtype)
    return result.to(up.dtype), logz


def segmented_attention(q, prefix_k, prefix_v, tail_k, tail_v, *,
                        reader: Reader | None = None, backend: Backend = "reference",
                        q_positions: torch.Tensor | None = None):
    """Shared prefix + private causal tail. No per-layer prefix reconstruction."""
    qp = q if reader is None else query_transform(q, reader)
    up, lp = attention_with_lse(qp, prefix_k, prefix_v, backend=backend)
    if reader is not None:
        up = value_transform(up, reader)
        # A constant key offset cancels within prefix softmax, but NOT in the
        # competition with the private tail. Restore it to prefix log Z.
        lp = lp + key_offset(q, reader, q.shape[-1] ** -0.5).to(lp.dtype)
    ut, lt = attention_with_lse(q, tail_k, tail_v, causal=True, backend=backend, q_positions=q_positions)
    return merge_segments(up, lp, ut, lt)[0]
