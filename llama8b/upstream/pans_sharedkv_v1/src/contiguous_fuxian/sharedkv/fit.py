"""Offline closed-form fitting. No backbone update and no backpropagation.

The RoPE-commuting component is a prior, not a claim that pretrained layers
are exactly equivalent. The residual and affine offsets are prefix-specific.
"""
from __future__ import annotations
import torch


def _double(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().to(device="cpu", dtype=torch.float64)
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError("Expected finite rank-2 observations")
    return x


def commuting_rope_fit(x, y, ridge: float = 1e-4):
    """Fit same-frequency complex scale/rotation in split-half layout.

    Each row-vector pair maps through [[a,-b],[b,a]]. Such blocks commute
    with every rotation of that frequency. There is no cross-frequency mix.
    """
    x, y = _double(x), _double(y)
    if x.shape != y.shape or x.shape[1] % 2:
        raise ValueError("Equal [tokens, even_head_dim] matrices required")
    d = x.shape[1]
    half = d // 2
    x0, x1 = x[:, :half], x[:, half:]
    y0, y1 = y[:, :half], y[:, half:]
    denom = (x0.square() + x1.square()).sum(0)
    lam = ridge * denom.mean().clamp_min(1e-12)
    # Small shrinkage towards the identity: useful as a prior, not mandatory.
    a = ((x0*y0 + x1*y1).sum(0) + lam) / (denom + lam)
    b = (x1*y0 - x0*y1).sum(0) / (denom + lam)
    result = torch.zeros(d, d, dtype=torch.float64)
    ids = torch.arange(half)
    result[ids, ids] = a
    result[ids+half, ids+half] = a
    result[ids, ids+half] = -b
    result[ids+half, ids] = b
    return result


def weighted_reduced_rank(x, y, output_metric, rank: int, ridge: float):
    """Minimize ||(X D-Y)L||^2 + lambda||D L||^2, rank(D)<=rank.

    output_metric = L L.T. The truncated SVD is performed AFTER whitening
    the regression inputs and weighting observable output directions.
    Unlike truncating D itself, this solves the stated reduced-rank problem.
    """
    x, y = _double(x), _double(y)
    d, o = x.shape[1], y.shape[1]
    if x.shape[0] != y.shape[0] or not 0 <= rank <= min(d, o) or ridge <= 0:
        raise ValueError("Invalid rank/ridge/observation shapes")
    if rank == 0:
        return torch.zeros(d, o, dtype=torch.float64)
    metric = _double(output_metric)
    if metric.shape != (o, o):
        raise ValueError("Output metric shape mismatch")
    chol_m = torch.linalg.cholesky(metric)
    gram = x.T @ x
    lam = ridge * (gram.trace()/d).clamp_min(1e-12)
    chol_g = torch.linalg.cholesky(gram + lam * torch.eye(d, dtype=x.dtype))
    z = torch.linalg.solve_triangular(chol_g, x.T @ y @ chol_m, upper=False)
    u, s, vh = torch.linalg.svd(z, full_matrices=False)
    z_r = (u[:, :rank] * s[:rank]) @ vh[:rank]
    dl = torch.linalg.solve_triangular(chol_g.T, z_r, upper=True)
    # D L = dl => L.T D.T = dl.T; no explicit matrix inverse.
    return torch.linalg.solve_triangular(chol_m.T, dl.T, upper=True).T


def fit_key_reader(source_k, target_k, query_rows, *, rank: int = 16,
                   ridge: float = 1e-3, mode: str = "rope_rr"):
    """Fit K_target ~= K_source @ A + bk for ONE physical KV head.

    query_rows includes every query head in that GQA group, not just the
    representative index head. Keys and queries must both be post-RoPE.
    """
    x, y, q = _double(source_k), _double(target_k), _double(query_rows)
    if x.shape != y.shape or q.shape[1] != x.shape[1]:
        raise ValueError("Key/query feature mismatch")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x-mx, y-my
    d = x.shape[1]
    metric = q.T @ q / max(1, q.shape[0])
    metric = metric / (metric.trace()/d).clamp_min(1e-12)
    metric = metric + ridge * torch.eye(d, dtype=torch.float64)
    if mode == "dense":
        base, use_rank = torch.zeros(d, d, dtype=x.dtype), d
    elif mode == "rope":
        base, use_rank = commuting_rope_fit(xc, yc, ridge), 0
    elif mode == "rope_rr":
        base, use_rank = commuting_rope_fit(xc, yc, ridge), rank
    else:
        raise ValueError(f"Unknown key mode {mode}")
    delta = weighted_reduced_rank(xc, yc-xc@base, metric, use_rank, ridge)
    a = base + delta
    bias = my - mx @ a
    residual = x @ a + bias - y
    weighted_error = ((residual @ metric) * residual).sum()
    weighted_signal = ((y @ metric) * y).sum().clamp_min(1e-30)
    return a.float(), bias.float(), {
        "key_query_metric_relative_rmse_fit": float((weighted_error/weighted_signal).sqrt()),
        "key_residual_rank": use_rank,
        "key_mode": mode,
        "key_matrix_spectral_norm": float(torch.linalg.matrix_norm(a, ord=2)),
    }


def affine_ridge(x, y, *, ridge: float = 1e-3, prior=None):
    """Affine regression with a small positive bias regularizer."""
    x, y = _double(x), _double(y)
    xa = torch.cat((x, torch.ones(x.shape[0], 1, dtype=x.dtype)), dim=-1)
    if prior is None:
        prior = torch.zeros(xa.shape[1], y.shape[1], dtype=x.dtype)
    return ridge_design(xa, y, ridge=ridge, prior=prior)


def ridge_design(design, target, *, ridge: float, prior=None):
    """Regression of an explicitly supplied design matrix (last column bias)."""
    x, y = _double(design), _double(target)
    if ridge <= 0 or x.shape[0] != y.shape[0]:
        raise ValueError("Invalid ridge or data shape")
    p = torch.zeros(x.shape[1], y.shape[1], dtype=x.dtype) if prior is None else _double(prior)
    if p.shape != (x.shape[1], y.shape[1]):
        raise ValueError("Prior shape mismatch")
    gram = x.T @ x
    lam = ridge * (gram.trace()/x.shape[1]).clamp_min(1e-12)
    regularizer = torch.eye(x.shape[1], dtype=x.dtype) * lam
    regularizer[-1, -1] *= 0.01
    beta = torch.linalg.solve(gram + regularizer, x.T @ y + regularizer @ p)
    return beta[:-1].float(), beta[-1].float()


def fit_value_from_attention(shared_conditional, teacher_conditional,
                             private_conditional, shared_mass, teacher_mass,
                             *, ridge=1e-3, prior=None):
    """Fit B,bv using actual attention outputs, for ONE physical KV head.

    Teacher mixed output = beta_t*U_t + (1-beta_t)*U_private.
    Design = [beta_s*U_s, beta_s], target subtracts student's private part.
    Therefore the fit respects prefix-vs-private normalization and the
    attention changes induced by the key reader. It is NOT raw V-only MSE.
    """
    us, ut, uv = map(_double, (shared_conditional, teacher_conditional, private_conditional))
    bs = shared_mass.detach().cpu().double().reshape(-1, 1)
    bt = teacher_mass.detach().cpu().double().reshape(-1, 1)
    if us.shape != ut.shape or us.shape != uv.shape or bs.shape != bt.shape or bs.shape[0] != us.shape[0]:
        raise ValueError("Attention regression observation shapes mismatch")
    design = torch.cat((bs * us, bs), dim=-1)
    target = bt*ut + (bs-bt)*uv
    return ridge_design(design, target, ridge=ridge, prior=prior)
