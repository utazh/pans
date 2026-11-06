"""Compact all-GQA selector-key index used only by ProMixed probing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


INDEX_FORMAT = "promixed-key-index-v1"
INT4_LEVELS = 7
DEFAULT_GROUP_SIZE = 32


def pack_signed_int4(values: Any) -> Any:
    """Pack even-width signed values in [-7, 7], low nibble first."""

    import numpy as np

    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] % 2:
        raise ValueError("signed INT4 values require a non-empty even last dimension")
    if np.any(array < -INT4_LEVELS) or np.any(array > INT4_LEVELS):
        raise ValueError("signed INT4 values must be in [-7, 7]")
    encoded = array.astype(np.int8, copy=False).astype(np.uint8) & 0x0F
    return (
        encoded[..., 0::2] | (encoded[..., 1::2] << 4)
    ).astype(np.uint8, copy=False)


def unpack_signed_int4(codes: Any, *, width: int) -> Any:
    """Unpack low-nibble-first signed INT4 codes into an int8 array."""

    import numpy as np

    packed = np.asarray(codes, dtype=np.uint8)
    if width <= 0 or width % 2 or packed.shape[-1] * 2 != width:
        raise ValueError("INT4 width must be positive, even, and match the codes")
    unpacked = np.empty((*packed.shape[:-1], width), dtype=np.int8)
    unpacked[..., 0::2] = (packed & 0x0F).astype(np.int8)
    unpacked[..., 1::2] = (packed >> 4).astype(np.int8)
    unpacked[unpacked >= 8] -= 16
    return unpacked


def quantize_symmetric_int4(
    values: Any, *, group_size: int | None = None
) -> tuple[Any, Any]:
    """Quantize [token, head, dim] keys with one FP16 scale per feature group."""

    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or not all(size > 0 for size in array.shape):
        raise ValueError("selector keys must have shape [token, head, dim]")
    if array.shape[-1] % 2:
        raise ValueError("INT4 selector head dimension must be even")
    if not np.isfinite(array).all():
        raise ValueError("selector keys must be finite")
    head_dim = int(array.shape[-1])
    normalized_group_size = head_dim if group_size is None else int(group_size)
    if normalized_group_size <= 0 or head_dim % normalized_group_size:
        raise ValueError(
            "INT4 group size must be positive and divide the head dimension"
        )
    grouped = array.reshape(
        *array.shape[:-1],
        head_dim // normalized_group_size,
        normalized_group_size,
    )
    scales = np.max(np.abs(grouped), axis=-1) / INT4_LEVELS
    safe_scales = np.where(scales > 0, scales, 1.0)
    quantized = np.rint(grouped / safe_scales[..., None])
    quantized = np.clip(quantized, -INT4_LEVELS, INT4_LEVELS).astype(np.int8)
    quantized[scales == 0] = 0
    return (
        pack_signed_int4(quantized.reshape(array.shape)),
        scales.astype(np.float16),
    )


def dequantize_int4_numpy(
    codes: Any,
    scales: Any,
    *,
    head_dim: int,
    group_size: int | None = None,
) -> Any:
    """Reference NumPy dequantizer used by the builder and unit tests."""

    import numpy as np

    unpacked = unpack_signed_int4(codes, width=head_dim).astype(np.float32)
    scale_array = np.asarray(scales, dtype=np.float16)
    normalized_group_size = head_dim if group_size is None else int(group_size)
    if normalized_group_size <= 0 or head_dim % normalized_group_size:
        raise ValueError(
            "INT4 group size must be positive and divide the head dimension"
        )
    group_count = head_dim // normalized_group_size
    if scale_array.shape != (*unpacked.shape[:-1], group_count):
        raise ValueError("INT4 scales do not match token/head dimensions")
    grouped = unpacked.reshape(
        *unpacked.shape[:-1], group_count, normalized_group_size
    )
    return (
        grouped * scale_array.astype(np.float32)[..., None]
    ).reshape(unpacked.shape)



def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(model_path: str | Path) -> dict[str, Any]:
    """Content identity, including model weights, configuration/RoPE and adapters."""
    root = Path(model_path).resolve()
    names = sorted(set(root.glob("*.safetensors")) | set(root.glob("pytorch_model*.bin"))
                   | set(root.glob("adapter_model*.bin")))
    if not names or not (root / "config.json").is_file():
        raise ValueError("model identity requires local config and weight files")
    names += [p for p in (root / "config.json", root / "adapter_config.json") if p.is_file()]
    return {"files_sha256": {p.name: file_sha256(p) for p in names}}


def source_task_mapping(source: str | Path, explicit_order: Sequence[str] | None = None) -> dict[str, int]:
    """Use persisted source registration; legacy stores require an explicit order."""
    marker = Path(source) / ".contiguous_fuxian_complete"
    payload = json.loads(marker.read_text()) if marker.is_file() else {}
    order = payload.get("registered_store_tasks")
    if order is not None and explicit_order is not None and list(order) != list(explicit_order):
        raise ValueError("explicit source task order conflicts with completion marker")
    order = order if order is not None else explicit_order
    if not order or len(set(order)) != len(order):
        raise ValueError("source registration missing: provide the original source_task_order")
    return {str(task): prefix_id for prefix_id, task in enumerate(order)}


@dataclass(frozen=True)
class QuantizedKeyTask:
    name: str
    directory: Path
    prefix_tokens: int
    layers: int
    kv_heads: int
    head_dim: int
    group_size: int


class QuantizedKeyIndex:
    """Validated reader for a disk-backed symmetric INT4 selector index."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"selector index manifest was not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != INDEX_FORMAT or int(payload.get("bits", 0)) != 4:
            raise ValueError("unsupported selector index format or bit width")
        self.manifest_path = manifest_path
        self.payload = payload
        self.selector_kv_head_ids = tuple(
            int(head) for head in payload.get("selector_kv_head_ids", ())
        )
        if (
            not self.selector_kv_head_ids
            or len(set(self.selector_kv_head_ids)) != len(self.selector_kv_head_ids)
            or any(head < 0 for head in self.selector_kv_head_ids)
        ):
            raise ValueError("selector index head IDs must be unique and non-negative")
        task_payload = payload.get("tasks")
        if not isinstance(task_payload, dict) or not task_payload:
            raise ValueError("selector index manifest has no tasks")
        self.tasks: dict[str, QuantizedKeyTask] = {}
        # Preload the compact index before request timing so selector calls do
        # not repeat filesystem reads or host-memory pinning.
        self._cpu_cache: dict[tuple[str, int], tuple[Any, Any, int]] = {}

        for task, entry in task_payload.items():
            if not isinstance(entry, dict):
                raise ValueError(f"invalid selector index task entry {task!r}")
            directory = (self.root / str(entry["directory"])).resolve()
            try:
                directory.relative_to(self.root)
            except ValueError as exc:
                raise ValueError(f"selector index task {task!r} escapes its root") from exc
            normalized = QuantizedKeyTask(
                name=str(task),
                directory=directory,
                prefix_tokens=int(entry["prefix_tokens"]),
                layers=int(entry["layers"]),
                kv_heads=int(entry["kv_heads"]),
                head_dim=int(entry["head_dim"]),
                group_size=int(
                    entry.get("group_size", payload.get("group_size", entry["head_dim"]))
                ),
            )
            if (
                normalized.prefix_tokens <= 0
                or normalized.layers <= 0
                or normalized.kv_heads != len(self.selector_kv_head_ids)
                or normalized.head_dim <= 0
                or normalized.head_dim % 2
                or normalized.group_size <= 0
                or normalized.head_dim % normalized.group_size
            ):
                raise ValueError(f"invalid selector index geometry for task {task!r}")
            self.tasks[str(task)] = normalized
    @property
    def preloaded_compressed_bytes(self) -> int:
        return sum(item[2] for item in self._cpu_cache.values())

    def task_is_preloaded(self, task: str) -> bool:
        entry = self.tasks.get(task)
        return entry is not None and all(
            (task, layer) in self._cpu_cache for layer in range(entry.layers)
        )

    def preload_task(self, task: str) -> int:
        """Load one task's compressed index into reusable host tensors."""

        import numpy as np
        import torch

        entry = self.tasks.get(task)
        if entry is None:
            raise KeyError(f"selector index does not contain task {task!r}")
        loaded_bytes = 0
        for layer in range(entry.layers):
            cache_key = (task, layer)
            if cache_key in self._cpu_cache:
                continue
            codes_path, scales_path = self.layer_paths(task, layer)
            codes = np.fromfile(codes_path, dtype=np.uint8).reshape(
                entry.prefix_tokens, entry.kv_heads, entry.head_dim // 2
            )
            scales = np.fromfile(scales_path, dtype=np.float16).reshape(
                entry.prefix_tokens,
                entry.kv_heads,
                entry.head_dim // entry.group_size,
            )
            codes_tensor = torch.from_numpy(codes)
            scales_tensor = torch.from_numpy(scales)
            if torch.cuda.is_available():
                codes_tensor = codes_tensor.pin_memory()
                scales_tensor = scales_tensor.pin_memory()
            compressed_bytes = (
                int(codes_tensor.numel()) * int(codes_tensor.element_size())
                + int(scales_tensor.numel()) * int(scales_tensor.element_size())
            )
            self._cpu_cache[cache_key] = (
                codes_tensor,
                scales_tensor,
                compressed_bytes,
            )
            loaded_bytes += compressed_bytes
        return loaded_bytes


    @property
    def bits(self) -> int:
        return 4

    def validate_task(
        self,
        task: str,
        *,
        prefix_tokens: int,
        layers: int,
        kv_heads: int,
        head_dim: int,
        token_hash: str | None = None,
    ) -> QuantizedKeyTask:
        entry = self.tasks.get(task)
        if entry is None:
            raise KeyError(f"selector index does not contain task {task!r}")
        expected = (prefix_tokens, layers, kv_heads, head_dim)
        actual = (entry.prefix_tokens, entry.layers, entry.kv_heads, entry.head_dim)
        if actual != expected:
            raise ValueError(
                f"selector index geometry for {task!r} is {actual}, expected {expected}"
            )
        metadata = self.payload["tasks"][task]
        if token_hash is not None and metadata.get("token_hash") != token_hash:
            raise ValueError(f"selector index prefix token identity mismatch for {task!r}")
        digests = metadata.get("files_sha256")
        if token_hash is not None and not digests:
            raise ValueError("bound selector index requires per-file SHA256 digests")
        code_bytes = prefix_tokens * kv_heads * (head_dim // 2)
        scale_bytes = (
            prefix_tokens * kv_heads * (head_dim // entry.group_size) * 2
        )
        for layer in range(layers):
            codes_path, scales_path = self.layer_paths(task, layer)
            for path, expected_bytes in (
                (codes_path, code_bytes),
                (scales_path, scale_bytes),
            ):
                if not path.is_file() or path.stat().st_size != expected_bytes:
                    raise ValueError(
                        f"selector index file {path} is missing or has the wrong size"
                    )
                if digests is not None and digests.get(path.name) != file_sha256(path):
                    raise ValueError(f"selector index file digest mismatch: {path}")
        return entry

    def validate_model(self, identity: dict[str, Any]) -> None:
        if self.payload.get("model_identity") != identity:
            raise ValueError("selector index model/adapter/config identity mismatch")

    def layer_paths(self, task: str, layer: int) -> tuple[Path, Path]:
        entry = self.tasks.get(task)
        if entry is None:
            raise KeyError(f"selector index does not contain task {task!r}")
        if not 0 <= layer < entry.layers:
            raise ValueError(f"selector index layer {layer} is outside task {task!r}")
        return (
            entry.directory / f"layer_{layer:02d}.codes.u8",
            entry.directory / f"layer_{layer:02d}.scales.f16",
        )

    def load_layer(
        self, task: str, layer: int, *, device: Any
    ) -> tuple[Any, int, str]:
        """Transfer compressed bytes and dequantize on the caller's stream."""

        import numpy as np
        import torch

        entry = self.tasks[task]
        cache_key = (task, layer)
        cached = self._cpu_cache.get(cache_key)
        if cached is None:
            codes_path, scales_path = self.layer_paths(task, layer)
            codes = np.fromfile(codes_path, dtype=np.uint8).reshape(
                entry.prefix_tokens, entry.kv_heads, entry.head_dim // 2
            )
            scales = np.fromfile(scales_path, dtype=np.float16).reshape(
                entry.prefix_tokens,
                entry.kv_heads,
                entry.head_dim // entry.group_size,
            )
            codes_tensor = torch.from_numpy(codes)
            scales_tensor = torch.from_numpy(scales)
            compressed_bytes = (
                codes_path.stat().st_size + scales_path.stat().st_size
            )
            source = "disk"
        else:
            codes_tensor, scales_tensor, compressed_bytes = cached
            source = "cpu"
        target = torch.device(device)
        if target.type == "cuda":
            if not codes_tensor.is_pinned():
                codes_tensor = codes_tensor.pin_memory()
                scales_tensor = scales_tensor.pin_memory()
            codes_tensor = codes_tensor.to(target, non_blocking=True)
            scales_tensor = scales_tensor.to(target, non_blocking=True)
        else:
            codes_tensor = codes_tensor.to(target)
            scales_tensor = scales_tensor.to(target)
        low = (codes_tensor & 0x0F).to(torch.int16)
        high = (codes_tensor >> 4).to(torch.int16)
        unpacked = torch.stack((low, high), dim=-1).reshape(
            entry.prefix_tokens, entry.kv_heads, entry.head_dim
        )
        unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
        grouped = unpacked.reshape(
            entry.prefix_tokens,
            entry.kv_heads,
            entry.head_dim // entry.group_size,
            entry.group_size,
        )
        keys = (
            grouped.to(torch.float16) * scales_tensor.unsqueeze(-1)
        ).reshape(entry.prefix_tokens, entry.kv_heads, entry.head_dim)
        return keys.contiguous(), compressed_bytes, source


def build_quantized_key_index(
    *,
    source_pcache_dir: str | Path,
    store_root: str | Path,
    tasks: Sequence[str],
    output_dir: str | Path,
    selector_kv_head_ids: Sequence[int],
    group_size: int = DEFAULT_GROUP_SIZE,
    source_task_order: Sequence[str] | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build an INT4 index from Pcache's raw per-layer selector-head files."""

    import numpy as np

    from .sparse_qwen_reprefill import read_store_info

    source = Path(source_pcache_dir).resolve()
    output = Path(output_dir).resolve()
    normalized_tasks = tuple(str(task).strip().lower() for task in tasks)
    head_ids = tuple(int(head) for head in selector_kv_head_ids)
    normalized_group_size = int(group_size)
    if normalized_group_size <= 0:
        raise ValueError("selector index group size must be positive")
    if not normalized_tasks or len(set(normalized_tasks)) != len(normalized_tasks):
        raise ValueError("selector index tasks must be non-empty and unique")
    if not head_ids or len(set(head_ids)) != len(head_ids) or any(head < 0 for head in head_ids):
        raise ValueError("selector index head IDs must be unique and non-negative")
    if not source.is_dir():
        raise FileNotFoundError(f"source Pcache directory was not found: {source}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"selector index output must be new or empty: {output}")
    mapping = source_task_mapping(source, source_task_order)
    if any(task not in mapping for task in normalized_tasks):
        raise ValueError("requested task is absent from source registration")
    marker_path = source / ".contiguous_fuxian_complete"
    marker = json.loads(marker_path.read_text()) if marker_path.is_file() else {}
    if tuple(marker.get("selector_kv_head_ids", ())) != head_ids:
        raise ValueError("source selector head IDs do not match requested mapping")
    if marker.get("impress_reorder_sha256") or marker.get("physical_layout", "plain-logical-token-order") != "plain-logical-token-order":
        raise ValueError("selector builder requires plain logical token order")
    bound_model = model_identity(model_path) if model_path is not None else None
    output.mkdir(parents=True, exist_ok=True)

    task_entries: dict[str, dict[str, Any]] = {}
    total_source_bytes = 0
    total_index_bytes = 0
    total_squared_error = 0.0
    total_squared_signal = 0.0
    total_values = 0
    max_abs_error = 0.0
    for task in normalized_tasks:
        prefix_id = mapping[task]
        if not task.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"unsafe selector index task name {task!r}")
        info = read_store_info(store_root, task)
        if info.head_dim % normalized_group_size:
            raise ValueError(
                f"selector index group size does not divide task {task!r} head dimension"
            )
        task_dir = output / task
        task_dir.mkdir(parents=True, exist_ok=False)
        file_digests: dict[str, str] = {}
        task_squared_error = 0.0
        task_squared_signal = 0.0
        task_values = 0
        task_max_error = 0.0
        for layer in range(info.layers):
            source_path = source / f"{prefix_id}_{layer}_None_k.npy"
            expected_bytes = info.prefix_tokens * len(head_ids) * info.head_dim * 2
            if not source_path.is_file() or source_path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"source selector file {source_path} is missing or has the wrong size"
                )
            values = np.memmap(
                source_path,
                dtype=np.float16,
                mode="r",
                shape=(info.prefix_tokens, len(head_ids), info.head_dim),
            )
            # The legacy source marker may lack task names. Even an explicit
            # mapping must agree with the prepared prefix contents, not just shape.
            metadata_path = Path(store_root) / task / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("dtype") != "bfloat16" or metadata.get("layout") != "token,kv_head,head_dim":
                raise ValueError("source verification requires token-major BF16 prefix keys")
            prefix_path = Path(store_root) / task / f"layer_{layer:02d}_key.bf16"
            raw = np.fromfile(prefix_path, dtype=np.uint16)
            if raw.size != info.prefix_tokens * info.kv_heads * info.head_dim:
                raise ValueError("prepared prefix key file has wrong size")
            expected = (raw.astype(np.uint32) << 16).view(np.float32).reshape(
                info.prefix_tokens, info.kv_heads, info.head_dim
            )[:, head_ids, :].astype(np.float16)
            if not np.array_equal(values, expected):
                raise ValueError(f"source prefix content/mapping mismatch for {task!r} layer {layer}")
            codes, scales = quantize_symmetric_int4(values, group_size=normalized_group_size)
            codes_path = task_dir / f"layer_{layer:02d}.codes.u8"
            scales_path = task_dir / f"layer_{layer:02d}.scales.f16"
            codes.tofile(codes_path)
            scales.tofile(scales_path)
            file_digests[codes_path.name] = file_sha256(codes_path)
            file_digests[scales_path.name] = file_sha256(scales_path)
            reconstructed = dequantize_int4_numpy(
                codes, scales, head_dim=info.head_dim,
                group_size=normalized_group_size,
            )
            source_values = np.asarray(values, dtype=np.float32)
            errors = reconstructed - source_values
            squared_error = float(np.square(errors, dtype=np.float64).sum())
            squared_signal = float(np.square(source_values, dtype=np.float64).sum())
            task_squared_error += squared_error
            task_squared_signal += squared_signal
            task_values += int(errors.size)
            task_max_error = max(task_max_error, float(np.max(np.abs(errors))))
            total_source_bytes += expected_bytes
            total_index_bytes += codes_path.stat().st_size + scales_path.stat().st_size
        total_squared_error += task_squared_error
        total_squared_signal += task_squared_signal
        total_values += task_values
        max_abs_error = max(max_abs_error, task_max_error)
        task_entries[task] = {
            "directory": task,
            "prefix_id": prefix_id,
            "token_hash": info.token_hash,
            "files_sha256": file_digests,
            "prefix_tokens": info.prefix_tokens,
            "layers": info.layers,
            "kv_heads": len(head_ids),
            "head_dim": info.head_dim,
            "group_size": normalized_group_size,
            "rmse": math.sqrt(task_squared_error / max(1, task_values)),
            "relative_rmse": math.sqrt(task_squared_error / max(task_squared_signal, 1e-30)),
            "max_abs_error": task_max_error,
        }

    manifest: dict[str, Any] = {
        "format": INDEX_FORMAT,
        "identity_version": 2,
        "model_identity": bound_model,
        "layout": "layer,token,selector_kv_head,head_dim",
        "source_task_mapping": mapping,
        "bits": 4,
        "quantization": "symmetric-groupwise-per-token-head",
        "group_size": normalized_group_size,
        "packing": "signed-int4-low-nibble-first",
        "source_dtype": "float16",
        "scale_dtype": "float16",
        "selector_kv_head_ids": list(head_ids),
        "source_pcache_dir": str(source),
        "source_bytes": total_source_bytes,
        "index_bytes": total_index_bytes,
        "compression_ratio": total_source_bytes / max(1, total_index_bytes),
        "rmse": math.sqrt(total_squared_error / max(1, total_values)),
        "relative_rmse": math.sqrt(total_squared_error / max(total_squared_signal, 1e-30)),
        "max_abs_error": max_abs_error,
        "tasks": task_entries,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    return manifest
