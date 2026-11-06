#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contiguous_fuxian.sharedkv_v27.frozen_control import export_frozen_r0
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
a=p.parse_args();b=export_frozen_r0(a.input,a.output)
print('Exported frozen reader with R=0; NOT quality-certified:',a.output)
