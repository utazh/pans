#!/usr/bin/env python3
"""v25 policy options -- unchanged published v24 training options."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--native-blocks',type=int,required=True);p.add_argument('--chunk-size',type=int,default=16)
 p.add_argument('train_args',nargs=argparse.REMAINDER);a=p.parse_args()
 args=a.train_args[1:] if a.train_args[:1]==['--'] else a.train_args
 if not args:raise ValueError('Pass v24 training arguments after --')
 from contiguous_fuxian.sharedkv_v25.guard import check
 from contiguous_fuxian.sharedkv_v25.policy import NativePolicy
 from contiguous_fuxian.sharedkv_v25.train import training_patch
 check(ROOT)
 original=sys.argv
 try:
  sys.argv=[sys.argv[0],*args]
  with training_patch(NativePolicy(a.native_blocks,a.chunk_size)) as trainer:trainer.main()
 finally:sys.argv=original
if __name__=='__main__':main()
