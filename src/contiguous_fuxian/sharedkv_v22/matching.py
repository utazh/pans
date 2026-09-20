"""Small exact assignment over GQA groups; no model or data assumptions hidden here."""
from __future__ import annotations
import hashlib
import itertools
import json
import math
from typing import Sequence
import torch

FORMAT = 'pans-headaligned-readers-v22'
BASE_V2 = 'b48c45d779ecde8ae960a6e4837d8b6673c57600'


def validate_permutation(p: Sequence[int], heads: int) -> tuple[int, ...]:
    p = tuple(p)
    if not 1 <= heads <= 8:
        raise ValueError('This small-assignment prototype supports 1..8 KV heads')
    if len(p) != heads or any(type(x) is not int for x in p) or sorted(p) != list(range(heads)):
        raise ValueError('head map must be a bijection: target KV head -> source KV head')
    return p


def choose_assignment(cost: torch.Tensor) -> dict:
    """Find the minimum SUM of measured edge costs; identity wins ties.

    The finite-grid minimum is an optimization fact, NOT a guarantee of lower
    holdout error, classification loss, or end-to-end latency.
    """
    c = torch.as_tensor(cost, dtype=torch.float64, device='cpu')
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not 1 <= c.shape[0] <= 8:
        raise ValueError('cost must be square with at most eight heads')
    if not torch.isfinite(c).all() or (c < -1e-10).any():
        raise ValueError('Cost entries must be finite and nonnegative')
    h = c.shape[0]
    identity = tuple(range(h))
    def score(p): return sum(float(c[i, p[i]]) for i in range(h))
    baseline = score(identity)
    best, best_value = identity, baseline
    for p in itertools.permutations(range(h)):
        value = score(p)
        if value < best_value - 1e-12:
            best, best_value = p, value
    return {'permutation': list(best), 'identity_cost': baseline, 'matched_cost': best_value,
            'assignment_set_size': math.factorial(h),
            'relative_cost_reduction': (baseline-best_value)/max(baseline, 1e-30),
            'identity_is_selected': best == identity}


def split_fit(rows: list, *, selection_fraction: float = 0.25, seed: int = 22):
    """Split by QUERY, never by head/token. No held-out validation is used for matching."""
    if len(rows) < 4 or not 0 < selection_fraction < 1:
        raise ValueError('At least four fit queries and a proper selection fraction are needed')
    ordered = sorted(rows, key=lambda r: hashlib.sha256(
        (str(seed)+':'+r[0]['query_hash']).encode()).hexdigest())
    n_select = min(len(rows)-2, max(1, int(round(len(rows)*selection_fraction))))
    return ordered[n_select:], ordered[:n_select]


def normalized_maps(maps: dict, consumers, heads: int) -> dict[int, tuple[int, ...]]:
    parsed = {int(k): validate_permutation(v, heads) for k, v in maps.items()}
    consumers = set(int(x) for x in consumers)
    if set(parsed) - consumers:
        raise ValueError('Head map attached to a non-consumer layer')
    return {lid: parsed.get(lid, tuple(range(heads))) for lid in sorted(consumers)}


def aligned_fingerprint(readers: dict, maps: dict) -> str:
    """Legacy-equivalent maps retain the old fingerprint; nonidentity maps are bound."""
    from contiguous_fuxian.sharedkv_v21.traces import fingerprint
    base = fingerprint(readers)
    nontrivial = {str(k): list(v) for k, v in sorted(maps.items()) if tuple(v) != tuple(range(len(v)))}
    if not nontrivial:
        return base
    encoded = json.dumps(nontrivial, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256((base+'|head-alignment-v22|'+encoded).encode()).hexdigest()
