"""v30: turn CAKE/LAVa statistics (runtime_data/v30_kvscore/<mode>_<task>.npz) into engine inputs for one budget.
  static  : configs/v30_sel/<method>_prefix_<task>_k<tag>.json        (fixed per-layer chunks, query-independent)
  oracle  : runtime_data/v30_oracle/<method>_<task>_k<tag>_<split>/<query_hash>.json  (per query; quality reference)
  profile : configs/v30_sel/<method>budget_<task>_k<tag>.json        (layer budgets only, online selector picks chunks)
Global budget: T = floor(ratio * NB * L + 0.5) chunks, the engine's exact-budget target. Layer split: CAKE P_l =
H_l^(1/tau1) V_l^(1/tau2); LAVa e_l. CAKE (tau1, tau2) from the official model2tau.json for qwen2.5-7b-instruct:
per-layer average kept tokens >= 362 (geometric midpoint of the 128 / 1024 settings) -> (1.5, 0.6), else (0.4, 0.6).
Usage: v30_select_build.py <static|oracle|profile> <cake|lava> <task> <budget_percent> [split]"""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import os
R = Path(os.environ.get('V30_ROOT', '/home/panzihang/src/prism_layerkv')); CHUNK = 16
MODEL = os.environ.get('V30_MODEL', '/data1/llm/Qwen/Qwen2.5-7B-Instruct')
# official CAKE model2tau.json: qwen2.5-7b-instruct 1024:(1.5,0.6) 128:(0.4,0.6); llama3.1-8b-128k 1024:(1.6,0.4) 128:(1.6,0.6)
TAU = {'1024': (1.6, 0.4), '128': (1.6, 0.6)} if 'Llama' in MODEL else {'1024': (1.5, 0.6), '128': (0.4, 0.6)}
kind, method, task, budget = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
split = sys.argv[5] if kind == 'oracle' else None
import os
TAG = os.environ.get('V30_TAG', '')   # QuALITY article sets: e.g. qf0_
ratio = budget / 100; tag = f'{budget:.4f}'.rstrip('0').rstrip('.').replace('.', 'p')

def weights(z, i=None):
    pick = (lambda a: a) if i is None else (lambda a: a[i])
    N = int(z['N'])
    if method == 'lava': return np.asarray(pick(z['lava_entropy']), float), None
    tau = TAU['1024'] if ratio * N >= 362 else TAU['128']
    return np.asarray(pick(z['disp']), float) ** (1 / tau[0]) * np.asarray(pick(z['var']), float) ** (1 / tau[1]), tau

def split_blocks(w, NB, T, floor_):
    """Integer per-layer chunk counts proportional to w, each in [floor_, NB], summing to T (largest remainder,
    excess above NB redistributed as in CAKE's adjust_budgets)."""
    w = np.maximum(w, 1e-30); share = w / w.sum() * T
    cnt = np.clip(np.floor(share).astype(int), floor_, NB)
    while cnt.sum() != T:
        if cnt.sum() < T:
            room = np.where(cnt < NB)[0]; i = room[np.argmax((share - cnt)[room])]; cnt[i] += 1
        else:
            room = np.where(cnt > floor_)[0]; i = room[np.argmin((share - cnt)[room])]; cnt[i] -= 1
    return cnt

def choose(chunk_scores, cnt, forced, N):
    sel = []
    for l in range(L):
        keep = list(forced); order = np.argsort(-chunk_scores[l], kind='stable')
        for c in order:
            if len(keep) >= cnt[l]: break
            if c not in keep: keep.append(int(c))
        assert len(keep) == cnt[l]
        sel.append(sorted(t for c in keep for t in range(c * CHUNK, min((c + 1) * CHUNK, N))))
    return sel

if kind in ('static', 'profile'):
    src = R / f'runtime_data/v30_kvscore/prefix_{TAG}{task}.npz'; z = np.load(src); L = int(z['cake_chunk'].shape[0])
    N, NB = int(z['N']), int(z['NB']); T = math.floor(ratio * NB * L + 0.5)
    w, tau = weights(z); forced = [int(c) for c in z['forced_chunks']]
    common = dict(method=method, mode='prefix', task=task, budget_percent=budget, prefix_tokens=N, chunks_per_layer=NB,
                  blocks_total=T, tau=tau, gamma=200.0 if method == 'cake' else None, window=32,
                  source=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
    if kind == 'static':
        cnt = split_blocks(w, NB, T, len(forced))
        sel = choose(z[f'{method}_chunk'], cnt, forced, N)
        dst = R / f'configs/v30_sel/{method}_prefix_{TAG}{task}_k{tag}.json'
        body = dict(common, layer_blocks=cnt.tolist(), forced_chunks=forced, selected_by_layer=sel)
    else:
        share = w / w.sum() * ratio * L                      # continuous ratios, mean = ratio
        for _ in range(100):                                 # cap at 1, redistribute, keep the mean exact
            over = share > 1
            if not over.any(): break
            excess = (share[over] - 1).sum(); share[over] = 1; free = ~over
            share[free] += excess * share[free] / share[free].sum()
        share = share * (ratio * L / share.sum())
        dst = R / f'configs/v30_sel/{method}budget_{TAG}{task}_k{tag}.json'
        body = dict(schema_version=1, model=MODEL, target_mean_ratio=ratio, layer_ratios=share.tolist(),
                    calibration=dict(common, note='layer budget only; chunks chosen by the online selector'))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if json.loads(dst.read_text()) != json.loads(json.dumps(body)): raise SystemExit(f'exists with different content: {dst}')
        print('skip (identical)', dst.name); raise SystemExit(0)
    dst.write_text(json.dumps(body))
    print(dst.name, 'blocks', T, 'layer min/max', (min(body['layer_blocks']), max(body['layer_blocks'])) if kind == 'static'
          else (round(min(body['layer_ratios']), 4), round(max(body['layer_ratios']), 4)), 'tau', tau)
else:
    src = R / f'runtime_data/v30_kvscore/oracle_{task}.npz'; z = np.load(src); L = int(z['cake_chunk'].shape[1])
    N, NB = int(z['N']), int(z['NB']); T = math.floor(ratio * NB * L + 0.5)
    out = R / f'runtime_data/v30_oracle/{method}_{task}_k{tag}_{split}'
    if out.exists():
        if len(list(out.glob('*.json'))) == len(z['qhash']): print('skip (complete)', out.name); raise SystemExit(0)
        raise SystemExit(f'partial oracle dir: {out}')
    out.mkdir(parents=True)
    for i, qh in enumerate(z['qhash']):
        w, tau = weights(z, i); cnt = split_blocks(w, NB, T, 1)
        sel = choose(z[f'{method}_chunk'][i], cnt, [], N)
        (out / f'{qh}.json').write_text(json.dumps(dict(method=method, mode='oracle', task=task, query_hash=str(qh), uid=str(z['uid'][i]),
                                                        prefix_tokens=N, blocks_total=T, tau=tau, layer_blocks=cnt.tolist(), selected_by_layer=sel)))
    print(out.name, len(z['qhash']), 'queries; blocks', T)
