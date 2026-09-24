"""Separate full-quality, deployable timing, and theoretical representation tables."""
from __future__ import annotations
import json,csv,math
from pathlib import Path
import numpy as np
from .common import *

def paired_times(base,other,seed=2801):
 if set(base)!=set(other) or len(base)<3:raise ValueError('At least three paired blocks required')
 keys=sorted(base);a=np.array([base[k] for k in keys],dtype=float);b=np.array([other[k] for k in keys],dtype=float)
 if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(),b.min())<=0:raise ValueError('Bad times')
 rng=np.random.default_rng(seed);ix=rng.integers(0,len(a),size=(10000,len(a)))
 boots=100*(b[ix].mean(1)/a[ix].mean(1)-1)
 return dict(blocks=len(a),baseline_ms=float(a.mean()),student_ms=float(b.mean()),
     delta_percent=float(100*(b.mean()/a.mean()-1)),exploratory_block_ci_percent=np.quantile(boots,[.025,.975]).tolist(),
     per_block_delta_percent=(100*(b/a-1)).tolist(),stable_speed_certified=False)

def analyze(manifest,output):
 from contiguous_fuxian.sharedkv_v27.quality import load_records,paired_quality
 m=json.loads(Path(manifest).read_text());root=Path(m['root']);out=Path(output)
 if out.exists():raise FileExistsError(out)
 quality=[];timing=[];missing=[]
 assert_hashes(m['input_sha256'])
 complete=[]
 for p in sorted((root/'jobs').glob('*.json')):
  rec=json.loads(p.read_text())
  if rec['status']=='complete':assert_hashes(rec['result_sha256']);complete.append(rec['job'])
  else:missing.append(dict(job=p.name,status=rec['status']))
 for c in m['cells']:
  jobs=[j for j in complete if j['task']==c['task'] and j['budget']==c['budget']]
  qs=[j for j in jobs if j['phase'] in ('quality','external_quality')]
  expected=['I','R4','R0']+[f'kvsharer_g{g}' for g in m['spec']['kvsharer_g']]+[f'clf_{rk}_{rv}_{d}' for rk,rv in m['spec']['ranks'] for d in ('fp32','bf16')]
  observed={j['arm'] for j in qs}
  for arm in expected:
   if arm not in observed:missing.append(dict(cell=c['id'],missing='quality:'+arm))
  native=next((j for j in qs if j['arm']=='I'),None)
  if native is None:missing.append(dict(cell=c['id'],missing='quality identity'));continue
  baseline=load_records(Path(native['output'])/'scored_records.jsonl')
  r4=next((j for j in qs if j['arm']=='R4'),None)
  for j in qs:
   filename='quality_records_v27.jsonl' if j['quality_only'] else 'scored_records.jsonl'
   row=paired_quality(baseline,load_records(Path(j['output'])/filename))
   row.update(cell=c['id'],arm=j['arm'],quality_only=j['quality_only'],quality_dataset='historical_full_split')
   quality.append(row)
   if j['arm']=='R0' and r4:
    pair=paired_quality(load_records(Path(r4['output'])/'scored_records.jsonl'),load_records(Path(j['output'])/filename))
    pair.update(cell=c['id'],arm='R0_vs_R4',quality_only=False,quality_dataset='historical_full_split');quality.append(pair)
  for condition in m['spec']['conditions']:
   grouped={arm:{} for arm in ('I','R4','R0')};uidsets={}
   for j in jobs:
    if j['phase']!='timing' or j['condition']!=condition:continue
    if j['quality_only']:raise ValueError('External bridge in timing table')
    rows=read_jsonl(Path(j['output'])/'scored_records.jsonl');ids=tuple(str(r['uid']) for r in rows)
    if j['block'] in uidsets and uidsets[j['block']]!=ids:raise ValueError('Unpaired timing query order')
    uidsets[j['block']]=ids;times=[float(r['ttft_ms']) for r in rows]
    grouped[j['arm']][j['block']]=float(np.mean(times))
   for target,reference in [('R4','I'),('R0','I'),('R0','R4')]:
    if len(grouped[reference])==m['spec']['blocks'] and len(grouped[target])==m['spec']['blocks']:
     r=paired_times(grouped[reference],grouped[target]);r.update(cell=c['id'],condition=condition,arm=target,reference=reference);timing.append(r)
    else:missing.append(dict(cell=c['id'],condition=condition,missing=target+'_vs_'+reference))
 out.mkdir(parents=True)
 from .resources import collect
 resources=collect(m)
 write_json(out/'analysis.json',dict(quality=quality,timing=timing,external_reference_resources=resources,missing_or_incomplete=missing,
       note='No official external baseline latency, no pooled cross-budget independent samples, no automatic certification.'))
 for name,data in [('quality',quality),('timing',timing),('external_theory_only',resources)]:
  if not data:continue
  keys=sorted(set().union(*(r.keys() for r in data)))
  with (out/(name+'.csv')).open('x',newline='',encoding='utf8') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
   for r in data:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
