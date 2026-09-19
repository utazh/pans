"""Collect unlabelled teacher queries on separate calibration requests.

This deliberately does extra projections, CPU copies and file I/O. NEVER use
its timings as performance measurements. Prefix payload is NOT dumped again.
"""
from __future__ import annotations
from pathlib import Path
import torch
from .core import apply_rope, attention_with_lse
from .io import TRACE_FORMAT, BASE_COMMIT, geometry, query_hash, save_new


class Collector:
    def __init__(self, trace_dir, max_query_samples=8):
        self.trace_dir = Path(trace_dir)
        self.max_query_samples = int(max_query_samples)
        if self.max_query_samples <= 0:
            raise ValueError("max_query_samples must be positive")

    def install(self, runner):
        original_greedy = runner.greedy_flexgen_completion
        def completion(*args, **kw):
            model, loader = kw["model"], kw["loader"]
            digest = query_hash(kw["query_token_ids"])
            task = loader._info.task
            outpath = self.trace_dir/task/f"{digest}.pt"
            if outpath.exists():  # Deduplicate warmup and repeated queries.
                return original_greedy(*args, **kw)
            records, handles = {}, []
            for lid, layer in enumerate(model.model.layers):
                def before(module, inputs, kwargs, lid=lid):
                    if lid in records:
                        return  # Only the initial query prefill, not label decode.
                    x = kwargs.get("hidden_states", inputs[0] if inputs else None)
                    if x is None or "position_embeddings" not in kwargs:
                        raise RuntimeError("Unsupported Qwen attention call signature")
                    cos, sin = kwargs["position_embeddings"]
                    b, t, _ = x.shape
                    d = module.head_dim
                    if b != 1:
                        raise ValueError("Calibration v1 requires batch=1")
                    with torch.no_grad():
                        q = apply_rope(module.q_proj(x).view(b, t, -1, d), cos, sin)
                        k = apply_rope(module.k_proj(x).view(b, t, -1, d), cos, sin)
                        v = module.v_proj(x).view(b, t, -1, d)
                        pos = (torch.tensor([t-1], device=x.device) if self.max_query_samples == 1 else
                               torch.linspace(0, t-1, min(t, self.max_query_samples), device=x.device).round().long().unique())
                        q_sample = q.index_select(1, pos)
                        ut, lt = attention_with_lse(q_sample, k, v, causal=True,
                                                   q_positions=pos, backend="reference")
                    records[lid] = {
                        "q": q_sample[0].detach().cpu(),
                        "tail_output": ut[0].detach().cpu(),
                        "tail_lse": lt[0].transpose(0, 1).detach().cpu(),
                        "query_positions": pos.cpu(),
                        "selected_tokens": torch.tensor(loader._selected_by_layer[lid], dtype=torch.long),
                    }
                handles.append(layer.self_attn.register_forward_pre_hook(before, with_kwargs=True))
            try:
                result = original_greedy(*args, **kw)
            finally:
                for handle in handles:
                    handle.remove()
            if len(records) != len(model.model.layers):
                raise RuntimeError("Incomplete calibration trace")
            if loader._selector_index is None:
                raise ValueError("Use the current K4 index so its validated model identity is available")
            save_new(outpath, {"format": TRACE_FORMAT, "base_commit": BASE_COMMIT,
                "task": task, "query_hash": digest, "token_hash": loader._info.token_hash,
                "prefix_tokens": loader.prefix_tokens, "geometry": geometry(model.config),
                "model_identity": loader._selector_index.payload["model_identity"],
                "period": int(kw["impress_selection_period_size"]),
                "keep_ratio": float(loader.keep_ratio), "layers": records})
            return result
        runner.greedy_flexgen_completion = completion
