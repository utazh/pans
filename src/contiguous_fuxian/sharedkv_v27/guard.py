"""Fail closed on the published interfaces actually inspected for this add-on."""
import hashlib
from pathlib import Path
from .common import BASE
CHECKS={
 'src/contiguous_fuxian/sharedkv_v25/runtime.py':'ba99cb18db1f20ce50b4388d61db036f5be7a3ad',
 'scripts/run_evidence_v26.py':'64076b8085e21ef0f8b84134b7a4b812fedfcad4',
 'scripts/run_round4.py':'ee2fa959b94ee8190797c93c49a6b2805d3015ff',
 'src/contiguous_fuxian/sharedkv/runtime.py':'944ee1cd079cd75cc4e1d7b44000b6f1c679ad2c',
 'src/contiguous_fuxian/sharedkv_v26/control.py':'ebb3cb1db45690ab3f40041fa54925faca0005e1',
 'scripts/v26_training_worker.py':'e37e87ba646d8009f555d2b939dedb472751fc96',
 'scripts/v26_common.py':'ad9552294bd73a95b843472db6009bbb97cb066e'}
def check(root):
 for name,expected in CHECKS.items():
  raw=(Path(root)/name).read_bytes();actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
  if actual!=expected:raise RuntimeError(f'{name}: interface differs from {BASE}; audit before changing guard')
