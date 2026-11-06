"""On-disk contracts and the exact raw-BF16 prefix layout used by pans."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

BASE_COMMIT = "e32c4a5605ca55d2904b33065d67a1d71667d47c"
FORMAT = "pans-readside-shared-prefix-v1"
TRACE_FORMAT = "pans-readside-calibration-v1"


def query_hash(ids) -> str:
    return hashlib.sha256(json.dumps([int(x) for x in ids], separators=(",", ":")).encode()).hexdigest()


def save_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    torch.save(data, path)


def load_safe(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def read_prefix(store_root, task, layer, kind):
    """Return the values actually seen by FP16-payload/BF16-compute pans.

    Existing store is raw BF16, token-major; not a NumPy .npy container.
    """
    root = Path(store_root)/task
    meta = json.loads((root/"metadata.json").read_text())
    if meta.get("dtype") != "bfloat16" or meta.get("layout") != "token,kv_head,head_dim":
        raise ValueError("This add-on requires the existing BF16 token-major store")
    n, h, d = (int(meta[x]) for x in ("prefix_tokens", "kv_heads", "head_dim"))
    path = root/f"layer_{layer:02d}_{kind}.bf16"
    if kind not in ("key", "value") or path.stat().st_size != n*h*d*2:
        raise ValueError(f"Invalid prefix tensor {path}")
    raw = np.fromfile(path, dtype=np.uint16).reshape(n, h, d)
    values = torch.from_numpy(raw).view(torch.bfloat16)
    values = values.to(torch.float16).to(torch.bfloat16).float()
    return values, meta


def validate_sources(sources, *, period):
    """One-hop ownership only; leaders exact; all consumers in one P8 window."""
    n = len(sources)
    if period <= 0 or not n:
        raise ValueError("Invalid layer/selection-period geometry")
    for layer, source in enumerate(sources):
        if not isinstance(source, int) or not 0 <= source < n:
            raise ValueError("Out-of-range physical source")
        if sources[source] != source:
            raise ValueError("Chained/cyclic ownership is forbidden")
        if layer//period != source//period:
            raise ValueError("Sharing cannot cross a selector window in v1")
        if layer % period == 0 and source != layer:
            raise ValueError("P8 selector leaders remain exact in v1")


def geometry(config):
    return {"layers": int(config.num_hidden_layers), "q_heads": int(config.num_attention_heads),
            "kv_heads": int(config.num_key_value_heads),
            "head_dim": int(getattr(config, "head_dim", config.hidden_size//config.num_attention_heads))}
