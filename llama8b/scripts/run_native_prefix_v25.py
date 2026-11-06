#!/usr/bin/env python3
"""v25 entry: options -- existing v23 options -- v21 options -- round4 options.

Cold certification and runtime warmup are delegated unchanged to published v23.
No current-query labels or baseline outputs are read during inference.
"""
from pathlib import Path
import argparse,json,runpy,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model-path');p.add_argument('--store-root');p.add_argument('--selector-index-dir')
 p.add_argument('--auto-probe-heads',action='store_true')
 p.add_argument('rest',nargs=argparse.REMAINDER);a=p.parse_args()
 args=a.rest[1:] if a.rest[:1]==['--'] else a.rest
 if args.count('--')<2:raise ValueError('Need v23 options -- v21 options -- round4 options')
 from contiguous_fuxian.sharedkv_v25.guard import check
 check(ROOT)
 from contiguous_fuxian.sharedkv_v21 import runtime as base
 from contiguous_fuxian.sharedkv_v25.runtime import Runtime
 import contiguous_fuxian.flexgen_qwen_reprefill as runner
 old_class,old_main,old_argv=base.Runtime,runner.main,sys.argv
 overrides={flag:v for flag,v in (('--model-path',a.model_path),('--store-root',a.store_root),('--selector-index-dir',a.selector_index_dir)) if v}
 if a.auto_probe_heads:
  if not a.model_path:raise ValueError('--auto-probe-heads requires explicit --model-path')
  from contiguous_fuxian.sharedkv_v25.geometry import inspect_config
  g=inspect_config(json.loads((Path(a.model_path)/'config.json').read_text()))
  overrides['--probe-query-heads']=','.join(map(str,g['probe_query_heads']))
  overrides['--selector-kv-head-ids']=','.join(map(str,g['selector_kv_head_ids']))
 def execute_main(*pa,**kw):
  for flag,v in overrides.items():
   if flag not in sys.argv:raise ValueError(f'Expected original runner argument: {flag}')
   sys.argv[sys.argv.index(flag)+1]=v
  return old_main(*pa,**kw)
 base.Runtime=Runtime;runner.main=execute_main
 try:
  sys.argv=[str(ROOT/'scripts/run_sharedkv_v23.py'),*args]
  runpy.run_path(str(ROOT/'scripts/run_sharedkv_v23.py'),run_name='__main__')
 finally:base.Runtime=old_class;runner.main=old_main;sys.argv=old_argv
if __name__=='__main__':main()
