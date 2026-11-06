#!/usr/bin/env python3
"""v29 UNIFIED row-restore backend (same code for every arm), inside the unchanged run_evidence_v26 chain.
  --engine native : prefix K/V rows from per-layer contiguous FP16 files assembled byte-identically from the pcache payload
                    (runtime_data/v29_rows_ssd/<task>/L<ll>_{k,v}.f16). With v21 --mode identity this is I (each layer reads
                    its own rows); with --mode shared + adapters it is our sharing (only physical SOURCE layers are read;
                    consumers reuse the source buffer; readers act inside attention as usual).
  --engine xkv    : xKV-CLF/P8 U rows (runtime_data/v29_xkv_ssd/<cell>), per-layer reconstruction + RoPE as in the v27 bridge.
Common mechanics (identical for all arms): when a P8 window's online selection is known, a 2-thread pool reads every needed
row set with one os.pread per run of consecutive rows into pinned host memory; resolve() waits, copies H2D and (xkv)
reconstructs. No cross-request application cache, no cache-score maintenance, no native pcache access. Request-dependent
reads are submitted only after the online selection and are inside the original TTFT boundary.
--row-cold: before every measured request (after --warmup) the engine's files are evicted (POSIX_FADV_DONTNEED) and
certified resident==0 via sharedkv_v23.cold.prepare_pages, outside TTFT. Usage:
  v29_rowengine_entry.py --engine native|xkv --data-dir D [--row-cold] --row-log L <run_evidence_v26 args...>"""
import json, os, runpy, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
def pop(flag, val=True):
    if flag not in sys.argv: return None
    i = sys.argv.index(flag); v = sys.argv[i + 1] if val else True; del sys.argv[i:i + (2 if val else 1)]; return v
ENGINE = pop('--engine'); DDIR = Path(pop('--data-dir')); COLD = bool(pop('--row-cold', False)); LOG = Path(pop('--row-log'))
WARMUP = int(sys.argv[sys.argv.index('--warmup') + 1]); TASK = sys.argv[sys.argv.index('--task') + 1]
assert ENGINE in ('native', 'xkv')
import numpy as np, torch
from contiguous_fuxian.sharedkv import runtime as v1
POOL = ThreadPoolExecutor(max_workers=2); FD = {}
def fd(path):
    if path not in FD: FD[path] = os.open(path, os.O_RDONLY)
    return FD[path]
def pread_rows(path, ids, row_bytes, np_dtype, shape_tail):
    """ids sorted unique int64; one pread per run of consecutive rows (no mmap: mapped pages cannot be evicted)."""
    cut = np.flatnonzero(np.diff(ids) != 1) + 1; f = fd(path)
    data = b''.join(os.pread(f, len(run) * row_bytes, int(run[0]) * row_bytes) for run in np.split(ids, cut))
    return torch.from_numpy(np.frombuffer(data, dtype=np_dtype).reshape(len(ids), *shape_tail).copy()).pin_memory()

if ENGINE == 'native':
    META = json.loads((ROOT / f'runtime_data/prefix_store/{TASK}/metadata.json').read_text())
    H, D, L = META['kv_heads'], META['head_dim'], META['layers']; ROW = H * D * 2
    FILES = [DDIR / f'L{l:02d}_{k}.f16' for l in range(L) for k in ('k', 'v')]
    def read_native(layer, ids):
        return {k: pread_rows(DDIR / f'L{layer:02d}_{k}.f16', ids, ROW, np.int16, (H, D)).view(torch.float16) for k in ('k', 'v')}
else:
    from contiguous_fuxian.sharedkv_v27.baseline_math import rotary
    RES = torch.load(DDIR / 'resident.pt', map_location='cpu', weights_only=False)
    FILES = [DDIR / g[k]['file'] for g in RES['groups'] for k in ('key', 'value')]
    def read_xkv(g, ids):
        return {k: pread_rows(DDIR / g[k]['file'], ids, g[k]['rank'] * 2, np.int16, (g[k]['rank'],)).view(torch.bfloat16) for k in ('key', 'value')}
    RIGHTS = {}
STATS = dict(engine=ENGINE, data_dir=str(DDIR), cold=COLD, warmup=WARMUP, requests=[])

def schedule_window(state, loader, start, end):
    x = state.__dict__.setdefault('_row', dict(fut={}, ids={}, bytes=0))
    if ENGINE == 'native':
        for l in range(start, end):
            s = state.sources[l]
            if loader._selected_by_layer[l] != loader._selected_by_layer[s]:
                raise RuntimeError('Shared consumers selected different token IDs; union mode is not implemented')
            if s in x['fut']: continue
            ids = np.unique(np.asarray(loader._selected_by_layer[s], dtype=np.int64))
            x['ids'][s] = ids; x['bytes'] += 2 * len(ids) * ROW; x['fut'][s] = POOL.submit(read_native, s, ids)
    else:
        for gi, g in enumerate(RES['groups']):
            if gi in x['fut'] or not any(start <= l < end for l in g['layers']): continue
            ids = np.unique(np.concatenate([np.asarray(loader._selected_by_layer[l], dtype=np.int64) for l in g['layers']]))
            x['ids'][gi] = ids; x['bytes'] += len(ids) * (g['key']['rank'] + g['value']['rank']) * 2
            x['fut'][gi] = POOL.submit(read_xkv, g, ids)

def _gpu(x, key, device):
    gpu = x.setdefault('gpu', {})
    if key not in gpu:
        host = x['fut'][key].result()
        gpu[key] = {k: v.to(device, non_blocking=True) for k, v in host.items()}
        gpu[key]['ids'] = torch.from_numpy(x['ids'][key]).to(device)
    return gpu[key]

def resolve(state, loader, layer, *, dtype, device):
    x = state._row
    if ENGINE == 'native':
        s = state.sources[layer]
        if s not in state.bank:
            g = _gpu(x, s, device); idx = torch.tensor(loader._selected_by_layer[s], device=device, dtype=torch.long)
            pos = torch.searchsorted(g['ids'], idx)
            k = g['k'].index_select(0, pos).to(device=device, dtype=dtype).contiguous()   # same conversion as RequestState.resolve
            v = g['v'].index_select(0, pos).to(device=device, dtype=dtype).contiguous()
            state.bank[s] = (k.unsqueeze(0), v.unsqueeze(0)); loader._loaded[s] = (k, v); state.resolve_calls += 1
        return state.bank[s]
    if layer in state.bank: return state.bank[layer]
    gi, local = RES['lookup'][layer]; grp = RES['groups'][gi]; g = _gpu(x, gi, device)
    rk = (str(device), gi, local)
    if rk not in RIGHTS: RIGHTS[rk] = {k: grp[k]['rights'][local].to(device) for k in ('key', 'value')}
    ids = loader._selected_by_layer[layer]; idx = torch.tensor(ids, device=device, dtype=torch.long)
    pos = torch.searchsorted(g['ids'], idx); n = len(ids); h, d = RES['geometry']['kv_heads'], RES['geometry']['head_dim']
    k = (g['key'].index_select(0, pos) @ RIGHTS[rk]['key']).reshape(n, h, d)
    k = rotary(k, idx + RES['position_offset'], theta=RES['rope_theta'])
    v = (g['value'].index_select(0, pos) @ RIGHTS[rk]['value']).reshape(n, h, d)
    kv = (k.to(dtype).unsqueeze(0), v.to(dtype).unsqueeze(0))
    state.bank[layer] = kv; loader._loaded[layer] = (kv[0][0], kv[1][0]); state.resolve_calls += 1
    return kv

v1.RequestState.schedule_window = schedule_window
v1.RequestState.resolve = resolve

import contiguous_fuxian.flexgen_qwen_reprefill as runner
from contiguous_fuxian.sharedkv_v23.cold import prepare_pages
_main = runner.main
def main(*a, **kw):
    inner = runner.greedy_flexgen_completion; n = {'i': 0}
    def completion(*args, **kwargs):
        n['i'] += 1; measured = n['i'] > WARMUP; cold = None
        if COLD and measured: cold = prepare_pages([str(p) for p in FILES], str(DDIR), 'ssd-cold')
        before = dict(v1.__dict__.get('_LAST_BYTES', {}))
        result = inner(*args, **kwargs)
        STATS['requests'].append(dict(index=n['i'], measured=measured,
                                      cold_resident_after=None if cold is None else cold['page_after']['resident'],
                                      cold_files=None if cold is None else cold['files_checked']))
        return result
    runner.greedy_flexgen_completion = completion
    return _main(*a, **kw)
runner.main = main
_sw = schedule_window
def schedule_logged(state, loader, start, end):
    _sw(state, loader, start, end)
    if end >= len(loader._selected_by_layer): STATS.setdefault('request_bytes', []).append(state._row['bytes'])
v1.RequestState.schedule_window = schedule_logged

sys.argv = [str(ROOT / 'scripts/run_evidence_v26.py')] + sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    POOL.shutdown(wait=True)
    LOG.write_text(json.dumps(STATS, indent=1))
