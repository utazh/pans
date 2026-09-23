#!/usr/bin/env python3
"""Numerical/quality integration only; NEVER report this path's TTFT or I/O."""
from pathlib import Path
import argparse,runpy,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def get(args,k,default=None):
 if args.count(k)>1:raise ValueError('Duplicate '+k)
 return args[args.index(k)+1] if k in args else default
p=argparse.ArgumentParser();p.add_argument('--representation',required=True);p.add_argument('rest',nargs=argparse.REMAINDER)
a=p.parse_args();args=a.rest[1:] if a.rest[:1]==['--'] else a.rest
if get(args,'--mode')!='identity' or '--diagnostic' not in args:raise ValueError('Need identity+diagnostic')
if get(args,'--cold','none')!='none' or '--nvtx' in args or get(args,'--profile','none')!='none':raise ValueError('Quality reference cannot be profiled/timed as a baseline')
if get(args,'--repair-values','native')!='native' or '--adapters' in args:raise ValueError('No learned reader/proxy in external reference')
from contiguous_fuxian.sharedkv_v27.common import load_tensor,json_new
from contiguous_fuxian.sharedkv_v27.guard import check
from contiguous_fuxian.sharedkv_v27.baseline_quality import install_quality_memory
check(ROOT);bundle=load_tensor(a.representation)
output=Path(get(args,'--output',''))
if not str(output) or output==Path('.'):raise ValueError('Explicit output directory required')
if output.exists():raise FileExistsError(output)
marker=Path(str(output)+'.quality_only.json')
json_new(marker,{'status':'started','representation':a.representation,'method':bundle['method'],
 'quality_only':True,'latency_valid':False,'resource_measurement_valid':False,
 'why':'Original native I/O is still executed; compressed factors are resident; not SSD xKV-SR performance.'})
# Also prohibit evaluation on baseline fitting queries (no file-size shortcut).
import contiguous_fuxian.sharedkv_v25.runtime as online
from contiguous_fuxian.sharedkv.io import query_hash
old_install=online.Runtime.install
bans=set(bundle['calibration_query_hashes'])
def install(self,runner):
 old_install(self,runner);previous=runner.greedy_flexgen_completion
 def call(*pa,**kw):
  if kw['impress_selection_period_size']!=bundle['period']:raise ValueError('Different selector period')
  cfg=kw['model'].config
  from contiguous_fuxian.sharedkv.io import geometry
  if geometry(cfg)!=bundle['geometry'] or cfg.model_type!='qwen2' or getattr(cfg,'rope_scaling',None) is not None or getattr(cfg,'use_sliding_window',False):
   raise ValueError('Baseline/model attention geometry differs')
  if float(getattr(cfg,'rope_theta',10000.))!=bundle['rope_theta']:
   raise ValueError('Baseline inverse/forward RoPE does not match runtime')
  if query_hash(kw['query_token_ids']) in bans:raise ValueError('Baseline FIT query used for evaluation')
  return previous(*pa,**kw)
 runner.greedy_flexgen_completion=call
online.Runtime.install=install
try:
 with install_quality_memory(bundle):
  sys.argv=[str(ROOT/'scripts/run_evidence_v26.py'),*args]
  runpy.run_path(str(ROOT/'scripts/run_evidence_v26.py'),run_name='__main__')
 # Produce a quality-only extract, not a disguised latency result.
 import json
 original_rows=output/'scored_records.jsonl'
 if original_rows.exists():
  with (output/'quality_records_v27.jsonl').open('x') as f:
   for line in original_rows.read_text().splitlines():
    r=json.loads(line);safe={k:v for k,v in r.items() if k in ('uid','task','prediction','correct','gold','label','answer')}
    safe.update(quality_only=True,latency_valid=False);f.write(json.dumps(safe)+'\n')
 json_new(Path(str(output)+'.quality_complete.json'),{'status':'complete','quality_only':True,'latency_valid':False})
finally:online.Runtime.install=old_install
