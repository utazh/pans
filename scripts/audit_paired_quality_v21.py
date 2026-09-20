#!/usr/bin/env python3
"""CSV schema: task,query_hash,correct (one row/query). No aggregate-only inference."""
from pathlib import Path
import argparse,csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from contiguous_fuxian.sharedkv_v21.audit import paired_quality
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--baseline",required=True);p.add_argument("--student",required=True)
p.add_argument("--output",required=True);p.add_argument("--margin-pp",type=float,default=1.)
a=p.parse_args()
if Path(a.output).exists():raise FileExistsError(a.output)
with open(a.baseline) as f:base=list(csv.DictReader(f))
with open(a.student) as f:student=list(csv.DictReader(f))
result=paired_quality(base,student,margin_pp=a.margin_pp)
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
