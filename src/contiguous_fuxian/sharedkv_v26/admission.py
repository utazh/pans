"""Opt-in stronger-baseline experiment: use resolved data during cache promotion.

This is NOT a new cross-layer compression method. Apply to identity AND shared.
Only complete raw-FP16 KV_Division chunks already returned by _resolve_entry are
eligible. Cache placement/scores/eviction decisions remain in ScoreUnit. A cache
entry receives an independent clone, never a view retaining the whole request.
No file is modified. No DONTNEED/mincore/cold-controller behavior is changed.

The host registry and D2D clone cost are INSIDE timing. Real GPU parity, heap,
residency and full-memory audits must pass before using this experimental path
as final evidence. CPU tests validate control flow, not GPU performance.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
import torch

@dataclass
class Entry:
    token: Any
    source: torch.Tensor
    key: bool

@dataclass
class ReuseRegistry:
    entries: dict[int, Entry] = field(default_factory=dict)
    registered: int = 0
    reused: int = 0
    reused_bytes: int = 0
    discarded_without_reuse: int = 0
    unsupported_to_calls: int = 0
    candidate_peak_entries: int = 0
    live_chunk_view_bytes: int = 0
    peak_chunk_view_bytes: int = 0
    remaining_at_request_end: int = 0

    def discard(self, token: Any) -> None:
        entry = self.entries.pop(id(token), None)
        if entry is not None:
            self.live_chunk_view_bytes -= entry.source.numel() * entry.source.element_size()
            self.discarded_without_reuse += 1

    def register(self, token: Any, tensor: torch.Tensor, key: bool) -> None:
        if getattr(token, 'device', None) != 'disk':
            return
        if not getattr(token, 'file_exist', False):
            raise ValueError('Only already-persisted chunks may bypass a disk restore')
        if bool(token.key_exist) == bool(token.value_exist) or bool(token.key_exist) != key:
            raise ValueError('Expected separate K/V chunks, not packed or mismatched data')
        if tensor.ndim != 3 or tuple(tensor.shape) != tuple(token.shape):
            raise ValueError('Do not admit a partial/wrong-shape chunk')
        if tensor.dtype != torch.float16 or str(token.dtype) != 'float16':
            raise ValueError('Only exact FP16 payload is supported; BF16 is not interchangeable')
        if id(token) in self.entries:
            raise ValueError('Duplicate unresolved cache token; speculative scheduling is unsupported')
        view = tensor.detach()
        self.entries[id(token)] = Entry(token, view, key)
        self.registered += 1
        self.live_chunk_view_bytes += view.numel() * view.element_size()
        self.peak_chunk_view_bytes = max(self.peak_chunk_view_bytes, self.live_chunk_view_bytes)
        self.candidate_peak_entries = max(self.candidate_peak_entries, len(self.entries))

    def register_resolved(self, loader: Any, layer_id: int, result: tuple) -> None:
        k, v, ids = result
        if ids.device.type != 'cpu' or ids.ndim != 1:
            raise ValueError('Expected the existing CPU physical_ids after _resolve_entry')
        if k.ndim != 3 or k.shape != v.shape or k.shape[0] != ids.numel():
            raise ValueError('Unexpected raw payload geometry')
        positions = ids.tolist()
        if positions != sorted(set(positions)):
            raise ValueError('Unsorted/duplicate physical positions are not supported')
        cache_layer = loader._pcache.cache[loader._prefix_id].layers[layer_id]
        if cache_layer.disk_type != 'KV_Division' or getattr(loader, 'impress_reordered', False):
            raise ValueError('Requires unreordered KV_Division storage')
        chunk_size, prefix_tokens = loader.chunk_size, loader.prefix_tokens
        cursor = 0
        while cursor < len(positions):
            chunk = positions[cursor] // chunk_size
            start, stop = chunk * chunk_size, min((chunk + 1) * chunk_size, prefix_tokens)
            count = stop - start
            if positions[cursor:cursor + count] != list(range(start, stop)):
                raise ValueError('Physical output must contain complete chunks, including short last chunk')
            self.register(cache_layer.key_tokens[chunk].token, k[cursor:cursor + count], True)
            self.register(cache_layer.value_tokens[chunk].token, v[cursor:cursor + count], False)
            cursor += count

    def promote(self, token: Any, device: Any, *, allow_cpu_test: bool = False) -> bool:
        entry = self.entries.get(id(token))
        if entry is None or token.device != 'disk':
            return False
        target = torch.device(device) if str(device) != 'disk' else None
        supported = target is not None and (target.type == 'cuda' or (allow_cpu_test and target.type == 'cpu'))
        if not supported:
            self.unsupported_to_calls += 1
            return False
        if not allow_cpu_test and entry.source.device != target:
            # torch.device('cuda') has no explicit index; compare using actual current index.
            if target.type != 'cuda' or entry.source.device.type != 'cuda' or (
                target.index is not None and entry.source.device.index != target.index
            ):
                raise ValueError('No cross-device promotion is implemented')
        if tuple(entry.source.shape) != tuple(token.shape):
            raise ValueError('Cache token metadata changed after prefetch')
        copied = entry.source.to(target).clone(memory_format=torch.contiguous_format)
        if entry.key:
            token.key = copied
        else:
            token.value = copied
        # Same bookkeeping as TokenCacheDisk.to. ScoreUnit still controls heap/capacity.
        token.device = device
        if token.pos_id is not None:
            token.prefix_cache.device_map[token.pos_id] = device
        self.reused += 1
        self.reused_bytes += copied.numel() * copied.element_size()
        self.entries.pop(id(token))
        self.live_chunk_view_bytes -= entry.source.numel() * entry.source.element_size()
        return True

    def finish(self) -> dict:
        self.remaining_at_request_end = len(self.entries)
        self.entries.clear()
        self.live_chunk_view_bytes = 0
        return {
            'registered_complete_tensor_chunks': self.registered,
            'promotion_reused_tensor_chunks': self.reused,
            'promotion_reused_payload_bytes': self.reused_bytes,
            'discarded_without_promotion_reuse': self.discarded_without_reuse,
            'unsupported_promotion_destinations': self.unsupported_to_calls,
            'registry_peak_entries': self.candidate_peak_entries,
            'peak_chunk_view_bytes_not_total_storage': self.peak_chunk_view_bytes,
            'remaining_entries_released_at_end': self.remaining_at_request_end,
            'note': 'Reused bytes are logical promotion payload, not measured SSD/H2D savings. '
                    'Views can retain larger backing tensors; measure total peak GPU memory.',
        }


@contextmanager
def admission_scope(loader: Any, registry: ReuseRegistry, *, vendor=None):
    """Single-request, single-threaded-inference installation; always restores APIs."""
    import sys
    if getattr(loader._config, 'cache_type', None) != 'CKLFU':
        raise ValueError('Only audited CKLFU placement is supported')
    if getattr(loader, 'defer_cache_score_updates', False):
        raise ValueError('Deferred admission is not part of this experiment')
    if any(getattr(loader, k, False) for k in (
        'impress_rolling_period_prefetch', 'impress_priority_prefetch', 'impress_value_ordered_prefetch'
    )) or getattr(loader, 'impress_period_prefetch_size', 1) != 1:
        raise ValueError('Speculative prefetch cannot share the promotion registry')
    vendor = vendor or sys.modules[type(loader._pcache).__module__]
    Token, Score = vendor.TokenCacheDisk, vendor.ScoreUnit
    old_to, old_update = Token.to, Score.update_score
    had_resolve = '_resolve_entry' in vars(loader)
    old_override = vars(loader).get('_resolve_entry')
    old_resolve = loader._resolve_entry

    def resolved(layer_id, entry):
        result = old_resolve(layer_id, entry)
        registry.register_resolved(loader, layer_id, result)
        return result

    def to(token, device):
        if registry.promote(token, device):
            return None
        return old_to(token, device)

    def update(control, item, new_score):
        try:
            return old_update(control, item, new_score)
        finally:
            # Discard even when policy left the chunk on disk/CPU; do not retain
            # all raw FP16 request payload until label scoring ends.
            registry.discard(item.token)

    loader._resolve_entry = resolved
    Token.to, Score.update_score = to, update
    try:
        yield
    finally:
        Token.to, Score.update_score = old_to, old_update
        if had_resolve:
            loader._resolve_entry = old_override
        else:
            delattr(loader, '_resolve_entry')
