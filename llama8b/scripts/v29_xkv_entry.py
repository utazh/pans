#!/usr/bin/env python3
"""xKV-CLF/P8-SSD launcher: SSD-native restoration of the shared xKV basis inside the unchanged run_evidence_v26 chain
(v21 --mode identity; our K4/P8 online selection protocol is kept). Native prefix payload is NEVER restored:
  schedule_window : when a P8 window's selection is known, a background thread reads, for each xKV group (4 layers) in the
                    window, the U rows of the union of the group's selected token IDs from the SSD files (key and value),
                    into pinned host memory.
  resolve         : waits for the group's rows, copies them to the GPU, and reconstructs each layer's K/V exactly as the
                    v27 quality bridge does (U[ids] @ R_layer in BF16, inverse-free RoPE on K at the token positions).
Cold condition (--xkv-cold): before every measured request (after --warmup), the U files are evicted with
POSIX_FADV_DONTNEED and certified resident==0 (sharedkv_v23.cold.prepare_pages), outside TTFT. Per-request read bytes
are logged to --xkv-log. Usage: v29_xkv_entry.py --xkv-dir D [--xkv-cold] --xkv-log L <run_evidence_v26 args...>"""
import json, runpy, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
def pop(flag, has_value=True):
    if flag not in sys.argv: return None
    i = sys.argv.index(flag); v = sys.argv[i + 1] if has_value else True; del sys.argv[i:i + (2 if has_value else 1)]; return v
XDIR = Path(pop('--xkv-dir')); COLD = bool(pop('--xkv-cold', False)); LOG = Path(pop('--xkv-log'))
WARMUP = int(sys.argv[sys.argv.index('--warmup') + 1])
import numpy as np, torch
from contiguous_fuxian.sharedkv import runtime as v1
from contiguous_fuxian.sharedkv_v27.baseline_math import rotary
RES = torch.load(XDIR / 'resident.pt', map_location='cpu', weights_only=False)
import os
# pread (not mmap): mapped pages cannot be evicted by POSIX_FADV_DONTNEED, which would invalidate the cold protocol.
FD = {}
for g in RES['groups']:
    for kind in ('key', 'value'):
        FD[(id(g), kind)] = os.open(XDIR / g[kind]['file'], os.O_RDONLY)
GPU_RIGHTS = {}
POOL = ThreadPoolExecutor(max_workers=2)
STATS = {'requests': []}

def _read(g, ids):
    # ids: sorted unique token ids; read each run of consecutive rows with one pread (16-token chunks are contiguous)
    cut = np.flatnonzero(np.diff(ids) != 1) + 1
    out = {}
    for kind in ('key', 'value'):
        r = g[kind]['rank']; row = r * 2; fd = FD[(id(g), kind)]; parts = []
        for run in np.split(ids, cut):
            parts.append(os.pread(fd, len(run) * row, int(run[0]) * row))
        rows = np.frombuffer(b''.join(parts), dtype=np.int16).reshape(len(ids), r)
        out[kind] = torch.from_numpy(rows.copy()).view(torch.bfloat16).pin_memory()
    return out

def schedule_window(state, loader, start, end):
    x = state.__dict__.setdefault('_xkv', dict(futures={}, unions={}, bytes=0))
    for gi, g in enumerate(RES['groups']):
        if gi in x['futures'] or not any(start <= l < end for l in g['layers']): continue
        ids = np.unique(np.concatenate([np.asarray(loader._selected_by_layer[l], dtype=np.int64) for l in g['layers']]))
        x['unions'][gi] = ids
        x['bytes'] += int(len(ids)) * (g['key']['rank'] + g['value']['rank']) * 2
        x['futures'][gi] = POOL.submit(_read, g, ids)

def resolve(state, loader, layer, *, dtype, device):
    if layer in state.bank: return state.bank[layer]
    x = state._xkv; gi, local = RES['lookup'][layer]; g = RES['groups'][gi]
    if 'gpu' not in x: x['gpu'] = {}
    if gi not in x['gpu']:
        host = x['futures'][gi].result()
        x['gpu'][gi] = {k: v.to(device, non_blocking=True) for k, v in host.items()}
        x['gpu'][gi]['union'] = torch.from_numpy(x['unions'][gi]).to(device)
    key = (str(device), gi, local)
    if key not in GPU_RIGHTS:
        GPU_RIGHTS[key] = {k: g[k]['rights'][local].to(device) for k in ('key', 'value')}
    ids = loader._selected_by_layer[layer]
    idx = torch.tensor(ids, device=device, dtype=torch.long)
    pos = torch.searchsorted(x['gpu'][gi]['union'], idx)
    n, (h, d) = len(ids), (RES['geometry']['kv_heads'], RES['geometry']['head_dim'])
    k = (x['gpu'][gi]['key'].index_select(0, pos) @ GPU_RIGHTS[key]['key']).reshape(n, h, d)
    k = rotary(k, idx + RES['position_offset'], theta=RES['rope_theta'])
    v = (x['gpu'][gi]['value'].index_select(0, pos) @ GPU_RIGHTS[key]['value']).reshape(n, h, d)
    kv = (k.to(dtype).unsqueeze(0), v.to(dtype).unsqueeze(0))
    state.bank[layer] = kv; loader._loaded[layer] = (kv[0][0], kv[1][0]); state.resolve_calls += 1
    return kv

v1.RequestState.schedule_window = schedule_window
v1.RequestState.resolve = resolve

import contiguous_fuxian.flexgen_qwen_reprefill as runner
from contiguous_fuxian.sharedkv_v23.cold import prepare_pages
FILES = [XDIR / g[k]['file'] for g in RES['groups'] for k in ('key', 'value')]
_main = runner.main
def main(*a, **kw):
    inner = runner.greedy_flexgen_completion; count = {'n': 0}
    def completion(*args, **kwargs):
        count['n'] += 1; measured = count['n'] > WARMUP; cold = None
        if COLD and measured: cold = prepare_pages([str(p) for p in FILES], str(XDIR), 'ssd-cold')
        result = inner(*args, **kwargs)
        st = v1._ACTIVE.get() if hasattr(v1, '_ACTIVE') else None
        STATS['requests'].append(dict(index=count['n'], measured=measured, cold_resident_after=None if cold is None else cold['page_after']['resident']))
        return result
    runner.greedy_flexgen_completion = completion
    return _main(*a, **kw)
runner.main = main

_orig_sw = schedule_window
def schedule_logged(state, loader, start, end):
    _orig_sw(state, loader, start, end)
    if start == 0: STATS.setdefault('bytes_per_request', []).append(None)
v1.RequestState.schedule_window = schedule_logged
_orig_res = resolve
def resolve_logged(state, loader, layer, *, dtype, device):
    kv = _orig_res(state, loader, layer, dtype=dtype, device=device)
    if layer == RES['geometry']['layers'] - 1 and STATS.get('bytes_per_request') and STATS['bytes_per_request'][-1] is None:
        STATS['bytes_per_request'][-1] = state._xkv['bytes']
    return kv
v1.RequestState.resolve = resolve_logged

sys.argv = [str(ROOT / 'scripts/run_evidence_v26.py')] + sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    POOL.shutdown(wait=True)
    LOG.write_text(json.dumps(dict(xkv_dir=str(XDIR), cold=COLD, warmup=WARMUP, **STATS), indent=1))
