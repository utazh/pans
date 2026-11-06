#!/usr/bin/env python3
"""v29 P1 mechanism ablations on the SAME source map / budget / checkpoint, without modifying existing files.
Replaces sharedkv_v21.ops.dispatch for consumer layers only, then runs the unchanged run_evidence_v26 chain.

  --variant direct    Same-graph direct sharing: consumers attend to the shared source K/V through the native
                      concatenated attention path; no reader tensors are applied (mathematically the unit reader,
                      but without executing identity matmuls). Not KVSharer (which has its own graph search).
  --variant explicit  Explicit reconstruction: materialize the consumer's approximate prefix K/V from the source,
                      K~ = K_s @ A + bk, V~ = V_s @ B + bv (per KV head, FP32 then cast to the KV dtype), then run the
                      native concatenated attention. Biases and joint normalization are preserved; the temporaries
                      live only for the layer call (allocator reuse), nothing is kept across layers.
Usage: v29_exec_variant.py --variant {direct,explicit} <run_evidence_v26 arguments...>"""
import runpy, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
i = sys.argv.index('--variant'); VARIANT = sys.argv[i + 1]; del sys.argv[i:i + 2]
if VARIANT not in ('direct', 'explicit'): raise SystemExit('unknown variant')
import torch
import contiguous_fuxian.sharedkv_v21.ops as ops
_orig = ops.dispatch
CALLS = {'native': 0, 'consumer': 0}

def dispatch(q, pk, pv, tk, tv, *, reader=None, packed=None, backend='reference', layout='split', annotate=False):
    if reader is None:
        CALLS['native'] += 1
        return _orig(q, pk, pv, tk, tv, backend=backend, layout=layout, annotate=annotate)
    CALLS['consumer'] += 1
    if VARIANT == 'direct':
        return _orig(q, pk, pv, tk, tv, backend=backend, layout=layout, annotate=annotate)
    if packed is not None:
        A, B, bk, bv = packed.A_transposed.transpose(-1, -2), packed.B, packed.bk_float, packed.bv
    else:
        A, B, bk, bv = reader.A, reader.B, reader.bk, reader.bv
    with ops.nvtx('sharedkv/explicit_reconstruct', annotate):
        kt = (torch.einsum('bnhd,hde->bnhe', pk.to(A.dtype), A) + bk.to(A.dtype)[None, None]).to(pk.dtype)
        vt = (torch.einsum('bnhd,hde->bnhe', pv.to(B.dtype), B) + bv.to(B.dtype)[None, None]).to(pv.dtype)
    return _orig(q, kt, vt, tk, tv, backend=backend, layout=layout, annotate=annotate)

ops.dispatch = dispatch
sys.argv = [str(ROOT / 'scripts/run_evidence_v26.py')] + sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    print(f'V29_EXEC_VARIANT {VARIANT} dispatch_calls {CALLS}', flush=True)
