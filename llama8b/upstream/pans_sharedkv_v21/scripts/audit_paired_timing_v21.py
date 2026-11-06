#!/usr/bin/env python3
"""Canonical CSV: block_id,task,query_hash,ttft_ms. Exclude warmups explicitly.

block_id identifies one repeated paired A/B experiment, not an individual query.
Requires exactly matching sets; bootstrap resamples whole repeated blocks.
It cannot verify that cache states, hardware load or timing boundaries matched.
"""
from pathlib import Path
import argparse,csv,json,math
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--baseline",required=True);p.add_argument("--student",required=True)
p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=19)
a=p.parse_args()
if Path(a.output).exists():raise FileExistsError(a.output)
def load(path):
    result={}
    with open(path) as f:
        for row in csv.DictReader(f):
            k=(row["block_id"],row["task"],row["query_hash"])
            if k in result:raise ValueError("Duplicate block/task/query")
            t=float(row["ttft_ms"])
            if not math.isfinite(t) or t<=0:raise ValueError("Invalid latency")
            result[k]=t
    return result
b,s=load(a.baseline),load(a.student)
if set(b)!=set(s) or not b:raise ValueError("Paired key sets differ")
blocks=sorted({k[0] for k in b});groups=[]
for block in blocks:
    keys=sorted(k for k in b if k[0]==block)
    groups.append((np.mean([b[k] for k in keys]),np.mean([s[k] for k in keys])))
base=np.array([x[0] for x in groups]);student=np.array([x[1] for x in groups])
result={"paired_blocks":len(blocks),"records_per_variant":len(b),
    "mean_block_baseline_ms":float(base.mean()),"mean_block_student_ms":float(student.mean()),
    "mean_block_delta_ms":float((student-base).mean()),
    "speedup_baseline_over_student":float(base.mean()/student.mean()),
    "relative_latency_change_pct":float(100*(student.mean()/base.mean()-1)),
    "conditional_block_bootstrap_95pct_delta_ms":None,
    "warning":"No hardware/cache/timing-boundary fairness certification; excludes warmups by input contract"}
if len(blocks)>=3:
    rng=np.random.default_rng(a.seed)
    delta=[float((student[idx]-base[idx]).mean()) for idx in rng.integers(0,len(blocks),(5000,len(blocks)))]
    result["conditional_block_bootstrap_95pct_delta_ms"]=[float(x) for x in np.quantile(delta,[.025,.975])]
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
