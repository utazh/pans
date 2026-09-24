#!/usr/bin/env python3
"""Frozen four-task matrix. Each execution is synchronous and explicitly requested."""
from pathlib import Path
import argparse,json,sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
 q=sub.add_parser('template');q.add_argument('--model-path',required=True);q.add_argument('--pcache-root',required=True);q.add_argument('--python');q.add_argument('--output',required=True)
 q=sub.add_parser('prepare');q.add_argument('--spec',required=True);q.add_argument('--output',required=True)
 q=sub.add_parser('compile-baselines');q.add_argument('--manifest',required=True);q.add_argument('--part',choices=['factors','kvsharer'],required=True);q.add_argument('--cells')
 q=sub.add_parser('run');q.add_argument('--manifest',required=True);q.add_argument('--stage',choices=['quality','baseline-quality','timing'],required=True);q.add_argument('--cells');q.add_argument('--execute',action='store_true')
 q=sub.add_parser('analyze');q.add_argument('--manifest',required=True);q.add_argument('--output',required=True)
 a=p.parse_args()
 from contiguous_fuxian.sharedkv_v28.common import write_json,check,assert_hashes
 if a.cmd=='template':
  from contiguous_fuxian.sharedkv_v28.protocol import make_spec
  write_json(a.output,make_spec(ROOT,a.model_path,a.pcache_root,a.python));return
 if a.cmd=='prepare':
  from contiguous_fuxian.sharedkv_v28.protocol import prepare
  prepare(json.loads(Path(a.spec).read_text()),a.output);return
 if a.cmd=='compile-baselines':
  m=json.loads(Path(a.manifest).read_text());check(m['spec']['repo']);assert_hashes(m['input_sha256'])
  os.environ['CUDA_VISIBLE_DEVICES']=str(m['spec']['gpu']);os.environ['OMP_NUM_THREADS']='4'
  if a.part=='factors':from contiguous_fuxian.sharedkv_v28.factors import compile_cell
  else:from contiguous_fuxian.sharedkv_v28.kvsharer import compile_cell
  chosen=a.cells.split(',') if a.cells else None
  if chosen and not set(chosen)<={c['id'] for c in m['cells']}:raise ValueError('Unknown cell ID')
  import torch,fcntl
  torch.set_num_threads(4)
  with (Path(m['root'])/'.foreground.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   for c in m['cells']:
    if chosen and c['id'] not in chosen:continue
    compile_cell(c,m['spec'],m['root'])
 elif a.cmd=='run':
  from contiguous_fuxian.sharedkv_v28.execute import execute_stage
  execute_stage(a.manifest,a.stage,a.cells.split(',') if a.cells else None,a.execute)
 else:
  from contiguous_fuxian.sharedkv_v28.analyze import analyze
  analyze(a.manifest,a.output)
if __name__=='__main__':main()
