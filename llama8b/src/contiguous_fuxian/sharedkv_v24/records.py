"""Compact, gold-label-free targets from the actual mean-continuation scorer."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import torch

FORMAT = 'pans-score-teacher-v24'
PLAN_FORMAT = 'pans-source-budget-plan-v24'
FIELDS = ('task','token_hash','prefix_tokens','geometry','period','keep_ratio','model_identity')


def digest_ids(ids):
    return hashlib.sha256(json.dumps([int(x) for x in ids],separators=(',',':')).encode()).hexdigest()


def json_digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save_new(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # x prevents accidental overwrites; inputs are generated tensors/primitives only.
    with path.open('xb') as f: torch.save(obj,f)


def validate(row):
    if row.get('format')!=FORMAT or row.get('teacher_sources')!=list(range(row['geometry']['layers'])):
        raise ValueError('A clean native teacher record is required; no shared teacher')
    if any(k in row for k in ('gold','gold_label','correct','answer')):
        raise ValueError('Gold labels are not part of the calibration contract')
    if row['query_hash']!=digest_ids(row['query_token_ids']) or not row['query_token_ids']:
        raise ValueError('Query hash/IDs mismatch')
    if row.get('score_rule')!='mean_continuation_logprob':raise ValueError('Unsupported score rule')
    labels=row['label_order']
    if len(labels)<2 or len(labels)!=len(set(labels)):
        raise ValueError('Need at least two distinct candidate labels')
    if set(labels)!=set(row['label_ids']) or set(labels)!=set(row['teacher_token_logprobs']):
        raise ValueError('Candidate/token-score keys differ')
    for label in labels:
        ids=row['label_ids'][label];lp=row['teacher_token_logprobs'][label]
        if not ids or len(ids)!=len(lp) or any(type(x)!=int or x<0 for x in ids):
            raise ValueError('Missing or invalid candidate tokens')
        if not all(math.isfinite(float(x)) for x in lp):raise ValueError('Nonfinite token target')
        actual=sum(map(float,lp))/len(lp)
        if abs(actual-float(row['teacher_scores'][label]))>1e-6:raise ValueError('Wrong length normalization')
    plan=row['selection_by_layer'];n=row['prefix_tokens']
    if len(plan)!=row['geometry']['layers']:raise ValueError('Missing layer plans')
    for ids in plan:
        if not ids or any(type(x)!=int or x<0 or x>=n for x in ids) or ids!=sorted(set(ids)):
            raise ValueError('Selected IDs must be sorted unique native positions')
    if row['selection_hash']!=json_digest(plan):raise ValueError('Selection hash mismatch')
    if row.get('split') not in ('fit','validation'):raise ValueError('Explicit fit/validation role required')


def load(root,task,role=None):
    root=Path(root); paths=sorted((root/task).glob('*.pt'))
    if not paths: raise FileNotFoundError(f'No v24 records in {root/task}')
    out={};first=None
    for path in paths:
        r=torch.load(path,map_location='cpu',weights_only=True);validate(r)
        if r['task']!=task or path.stem!=r['query_hash']:raise ValueError('Record filename/task mismatch')
        if role and r['split']!=role:raise ValueError('Wrong data split role')
        if first is None:first=r
        for k in FIELDS:
            if r[k]!=first[k]:raise ValueError(f'Mixed calibration identity: {k}')
        if r['query_hash'] in out:raise ValueError('Duplicate query record')
        out[r['query_hash']]=r
    return out


def assert_disjoint(fit,val):
    if set(fit)&set(val):raise ValueError('Fit/validation query leakage')
    f=next(iter(fit.values()));v=next(iter(val.values()))
    for key in FIELDS:
        if f[key]!=v[key]:raise ValueError(f'Fit/validation protocol mismatch: {key}')


def validate_sources(sources,period):
    n=len(sources)
    if not n or type(period)!=int or period<=0:raise ValueError('Invalid geometry')
    for lid,s in enumerate(sources):
        if type(s)!=int or not 0<=s<n or sources[s]!=s:raise ValueError('One-hop owned sources only')
        if lid//period!=s//period:raise ValueError('Cannot share across selector windows')
        if lid%period==0 and lid!=s:raise ValueError('Keep selector leaders native')


def assert_plan(row,sources):
    validate_sources(sources,row['period'])
    if len(sources)!=row['geometry']['layers']:raise ValueError('Wrong source-map length')
    for lid,s in enumerate(sources):
        if row['selection_by_layer'][lid]!=row['selection_by_layer'][s]:
            raise ValueError(f'Unequal selected IDs at layer {lid} source {s}; no union/truncation allowed')
