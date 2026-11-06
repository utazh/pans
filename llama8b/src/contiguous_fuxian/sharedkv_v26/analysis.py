"""Paired reports preserve the old conservative rule; no automatic promotion."""
from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np
from scipy.stats import beta

def records(path):
    out = {}
    for line in Path(path).read_text(encoding='utf8').splitlines():
        if not line.strip(): continue
        r = json.loads(line); key = (r.get('task',''), str(r['uid']))
        if key in out: raise ValueError('Duplicate task/UID: do not pool repetitions as new questions')
        if type(r['correct']) is not bool: raise ValueError('correct must be boolean')
        out[key] = r
    if not out: raise ValueError('Empty results')
    return out

def paired_quality(a, b, *, margin=.01, alpha=.025):
    if set(a) != set(b): raise ValueError('Exact pairing required; no silent intersection')
    if not 0 < margin < 1 or not 0 < alpha < .5: raise ValueError('Bad preregistered values')
    for key in a:
        if a[key].get('query_hash') and b[key].get('query_hash') and a[key]['query_hash'] != b[key]['query_hash']:
            raise ValueError('Same UID has a different query')
    n = len(a)
    h = sum(a[k]['correct'] and not b[k]['correct'] for k in a)
    g = sum(not a[k]['correct'] and b[k]['correct'] for k in a)
    lo = 0. if g == 0 else float(beta.ppf(alpha/2,g,n-g+1))
    hi = 1. if h == n else float(beta.ppf(1-alpha/2,h+1,n-h))
    return {'n_unique_queries': n, 'baseline_correct':sum(x['correct'] for x in a.values()),
            'student_correct':sum(x['correct'] for x in b.values()),
            'harmful':h,'beneficial':g,'net_delta_pp':100*(g-h)/n,
            'discordant_fraction':(h+g)/n,'conservative_lower_pp':100*(lo-hi),
            'margin_pp':100*margin,'alpha_per_route':alpha,
            'legacy_bound_passes':lo-hi >= -margin,
            'status':'Analysis only. Original failed preregistration is not overwritten; '
                     'seed/route decisions must be predeclared. Same queries across seeds are not independent.'}

def paired_time_blocks(blocks, *, draws=10000, seed=2601):
    """Each block is already a matched independent run unit; never flatten to N requests."""
    if len(blocks) < 4: raise ValueError('At least four paired blocks required; more are preferable')
    means=[]; ns=[]; all_unique=set()
    per=[]
    for r in blocks:
        a,b=r['baseline'],r['student']
        if set(a)!=set(b) or not a: raise ValueError('Mismatched timing queries')
        av=np.array([float(a[k]['ttft_ms']) for k in sorted(a)])
        bv=np.array([float(b[k]['ttft_ms']) for k in sorted(a)])
        if not np.isfinite(av).all() or not np.isfinite(bv).all() or min(av.min(),bv.min())<=0:
            raise ValueError('Invalid timings')
        means.append((av.mean(),bv.mean()));ns.append(len(a));all_unique.update(a)
        per.append({'block':r['block'],'n':len(a),'baseline_ms':float(av.mean()),
                    'student_ms':float(bv.mean()),'delta_pct':float(100*(bv.mean()/av.mean()-1))})
    means=np.asarray(means);ns=np.asarray(ns);rng=np.random.default_rng(seed)
    bs=[]
    for _ in range(draws):
        ix=rng.integers(0,len(means),size=len(means))
        x=np.average(means[ix],axis=0,weights=ns[ix]);bs.append(100*(x[1]/x[0]-1))
    x=np.average(means,axis=0,weights=ns)
    return {'paired_run_blocks':len(blocks),'unique_query_keys':len(all_unique),
            'request_pairs_including_repeats':int(ns.sum()),'baseline_ms':float(x[0]),
            'student_ms':float(x[1]),'delta_pct':float(100*(x[1]/x[0]-1)),
            'block_bootstrap95_pct':np.quantile(bs,[.025,.975]).tolist(),
            'per_block':per,'note':'Exploratory run-block uncertainty; not quality NI, '
                                  'not GPU causal decomposition. Temporal blocks can remain correlated.'}
