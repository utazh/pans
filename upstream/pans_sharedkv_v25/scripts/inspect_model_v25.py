#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from contiguous_fuxian.sharedkv_v25.geometry import inspect_config
p=argparse.ArgumentParser();p.add_argument('--model-path',required=True);a=p.parse_args()
print(json.dumps(inspect_config(json.loads((Path(a.model_path)/'config.json').read_text())),indent=2))
