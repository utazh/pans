"""Fail closed on the published interfaces actually inspected for this add-on."""
import hashlib
from pathlib import Path
from .common import BASE
CHECKS={
 'src/contiguous_fuxian/sharedkv_v25/runtime.py':'4e028ff638b54f6261f8ac1d8b8f63764020942a',
 'scripts/run_evidence_v26.py':'64076b8085e21ef0f8b84134b7a4b812fedfcad4',
 'scripts/run_round4.py':'ee2fa959b94ee8190797c93c49a6b2805d3015ff',
 'src/contiguous_fuxian/sharedkv/runtime.py':'42b322dd174bd3115b2dc1d0a921b5f28185d346',
 'src/contiguous_fuxian/sharedkv_v26/control.py':'ebb3cb1db45690ab3f40041fa54925faca0005e1',
 'scripts/v26_training_worker.py':'e37e87ba646d8009f555d2b939dedb472751fc96',
 'scripts/v26_common.py':'ad9552294bd73a95b843472db6009bbb97cb066e'}
def check(root):
 for name,expected in CHECKS.items():
  raw=(Path(root)/name).read_bytes();actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
  if actual!=expected:raise RuntimeError(f'{name}: interface differs from {BASE}; audit before changing guard')
