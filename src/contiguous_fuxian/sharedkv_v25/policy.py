"""Disjoint, unchanged-token-budget prefix partitioning.

The native region is the last R *selected* physical blocks, not an absolute
sliding window. It does not add dropped tokens and is not DeepSeek SWA replay.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

FORMAT = 'pans-shared-prefix-native-suffix-v25'
BASE = 'a6be130c921024a4cd358ccf5841e4a46f547ae2'

@dataclass(frozen=True)
class NativePolicy:
    blocks: int = 4
    chunk_size: int = 16
    rule: str = 'last_selected_blocks'

    def __post_init__(self):
        if type(self.blocks) is not int or self.blocks < 0:
            raise ValueError('Native block budget must be a nonnegative integer')
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError('Invalid chunk size')
        if self.rule != 'last_selected_blocks':
            raise ValueError('Only contiguous suffix in selected storage is implemented')

    def state(self):
        return dict(blocks=self.blocks, chunk_size=self.chunk_size, rule=self.rule)

@dataclass(frozen=True)
class Partition:
    cut: int
    shared_ids: tuple[int, ...]
    native_ids: tuple[int, ...]


def partition(ids: Sequence[int], prefix_tokens: int, policy: NativePolicy) -> Partition:
    values = tuple(ids)
    if not values or prefix_tokens <= 0:
        raise ValueError('Nonempty prefix selection required')
    if any(type(i) is not int or i < 0 or i >= prefix_tokens for i in values):
        raise ValueError('Invalid original token ID')
    if list(values) != sorted(set(values)):
        raise ValueError('Selection must be sorted and unique')
    block_ids = sorted({i // policy.chunk_size for i in values})
    # Exact physical chunks, including the legitimate short final chunk only.
    expected = tuple(i for c in block_ids for i in range(c*policy.chunk_size,
                    min((c+1)*policy.chunk_size, prefix_tokens)))
    if values != expected:
        raise ValueError('Partial selected chunks are not supported; do not round budget silently')
    if policy.blocks == 0:
        return Partition(len(values), values, ())
    if policy.blocks >= len(block_ids):
        raise ValueError('Repair budget must leave at least one shared block')
    boundary = block_ids[-policy.blocks] * policy.chunk_size
    cut = next(i for i, token in enumerate(values) if token >= boundary)
    return Partition(cut, values[:cut], values[cut:])


def payload_account(plan: Sequence[Sequence[int]], sources: Sequence[int], prefix_tokens: int,
                    kv_heads: int, head_dim: int, policy: NativePolicy, element_bytes: int = 2):
    if len(plan) != len(sources) or not plan or min(kv_heads,head_dim,element_bytes) <= 0:
        raise ValueError('Invalid accounting geometry')
    for l, s in enumerate(sources):
        if type(s) is not int or not 0 <= s < len(sources) or sources[s] != s:
            raise ValueError('One-hop physical ownership required')
        if list(plan[l]) != list(plan[s]):
            raise ValueError('Unequal source/consumer IDs; no union supported')
    byte_per_token = 2 * kv_heads * head_dim * element_bytes
    logical = sum(map(len,plan)) * byte_per_token
    base = sum(len(plan[s]) for s in set(sources)) * byte_per_token
    repair = sum(len(partition(plan[l],prefix_tokens,policy).native_ids)
                 for l,s in enumerate(sources) if l != s) * byte_per_token
    return dict(logical_selected_payload_bytes=logical, shared_base_payload_bytes=base,
                native_repair_payload_bytes=repair, total_selected_payload_bytes=base+repair,
                selected_payload_reduction_fraction=1-(base+repair)/logical,
                full_selected_source_count=len(set(sources)),
                partial_native_layer_count=sum(l != s and policy.blocks > 0 for l,s in enumerate(sources)),
                note='Includes full owner selections and consumer native suffixes; excludes readers/workspace/index. Not hardware I/O.')
