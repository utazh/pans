#!/usr/bin/env python3
"""Explicit compatibility conversion; rejects every nonidentity head map."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from contiguous_fuxian.sharedkv.io import FORMAT,load_safe,save_new
from contiguous_fuxian.sharedkv_v22.matching import FORMAT as V22_FORMAT
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
b=load_safe(a.input)
if b['format']!=V22_FORMAT:raise ValueError('Expected a published v22-format bundle')
h=b['geometry']['kv_heads']
if any(list(pi)!=list(range(h)) for pi in b.get('head_maps',{}).values()):
    raise ValueError('Nonidentity head routing cannot be discarded')
b['converted_from_format']=b['format'];b['format']=FORMAT;b.pop('head_maps',None)
b['diagnostic_only']=True;b['deployment_status']='compatibility_candidate_not_accepted';b['end_to_end_validated']=False
save_new(a.output,b)
print('Converted identity map only; reader tensors and calibration bans retained. Run parity check.')
