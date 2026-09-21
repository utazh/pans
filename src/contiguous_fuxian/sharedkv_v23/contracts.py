from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import torch

BASE = 'fddabb7015c66c870a11810069bfbe20d5a34bbf'
TRACE = 'pans-group-full-sequence-v23'
IDENTITY_FIELDS = ('task','token_hash','prefix_tokens','geometry','period','keep_ratio','model_identity')

@dataclass(frozen=True)
class Group:
    start: int
    end: int                 # exclusive
    source: int
    native: tuple[int, ...] = ()  # exact exceptions, e.g. a larger-budget P8 leader

    def validate(self, layers: int, period: int) -> None:
        if not 0 <= self.start < self.end <= layers or period < 1:
            raise ValueError('Invalid half-open group range')
        if len(set(self.native)) != len(self.native) or any(not self.start <= i < self.end for i in self.native):
            raise ValueError('Invalid native exception list')
        if not self.start <= self.source < self.end:
            raise ValueError('Source must belong to group')
        if self.start // period != (self.end - 1) // period:
            raise ValueError('A group cannot cross a selection window')
        for lid in self.consumers:
            if lid % period == 0:
                raise ValueError('Selector leaders must retain native prefix KV')

    @property
    def consumers(self):
        return [i for i in range(self.start, self.end) if i != self.source and i not in self.native]


def source_map(groups, layers=28, period=8):
    out=list(range(layers)); occupied=set()
    for group in groups:
        group.validate(layers,period)
        members=set(range(group.start,group.end))
        if members & occupied: raise ValueError('Overlapping groups')
        occupied |= members
        for lid in group.consumers:out[lid]=group.source
    return out


def tensor_fingerprint(readers):
    """Match v21 fingerprint: use its implementation, not an alternative digest."""
    from contiguous_fuxian.sharedkv_v21.traces import fingerprint
    return fingerprint(readers)


def load_dataset(root,task):
    result={}
    for p in sorted((Path(root)/task).glob('*.pt')):
        row=torch.load(p,map_location='cpu',weights_only=True)
        if row.get('format') != TRACE:raise ValueError(f'Not a full-sequence group trace: {p}')
        h=row['query_hash']
        if h in result:raise ValueError('Duplicate query trace')
        result[h]=row
    if not result:raise ValueError(f'No group traces in {root}/{task}')
    return result


def validate_pair(teacher,student,group):
    for k in IDENTITY_FIELDS:
        if teacher[k] != student[k]:raise ValueError(f'Paired trace mismatch: {k}')
    for x in (teacher,student):
        if x['group_start']!=group.start or x['group_end']!=group.end:
            raise ValueError('Capture a separate directory for each exact group range')
    if teacher['query_hash']!=student['query_hash'] or teacher['query_token_ids']!=student['query_token_ids']:
        raise ValueError('Mismatched actual query')
    if teacher['selection_by_layer']!=student['selection_by_layer']:
        raise ValueError('Calibration needs teacher-ID replay; do NOT disable selector computation')
    tp={tuple(v['suffix_ids']) for v in teacher['variants']}
    sp={tuple(v['suffix_ids']) for v in student['variants']}
    if tp!=sp:raise ValueError('Teacher/student label-prefix paths differ')
    for lid in group.consumers:
        if student['selection_by_layer'][lid]!=student['selection_by_layer'][group.source]:
            raise ValueError('Shared group selected unequal token IDs')


def selected_variants(row, include_suffix=True):
    vs=row['variants']
    return vs if include_suffix else [v for v in vs if not v['suffix_ids']]


def write_new(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    torch.save(data,p)
