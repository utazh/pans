"""Pinned Git blob identities verified through the GitHub connector."""
from pathlib import Path
import hashlib
from .policy import BASE
CHECKS={
 'src/contiguous_fuxian/sharedkv_v24/train.py':'7a7acc76c7d2ff26079665bb06480420c2bda3d9',
 'src/contiguous_fuxian/sharedkv_v24/replay.py':'d6b12864dd41b29256fb43f9dc609f1baad01c8a',
 'src/contiguous_fuxian/sharedkv_v24/faithful.py':'b50b810bed322465083f16a692cf61c52f5da0d2',
 'src/contiguous_fuxian/sharedkv_v24/records.py':'242a20032d77b8254f59f8c9d0e725b0c4509bbb',
 'src/contiguous_fuxian/sharedkv_v24/topology.py':'81c5560daf1a2fece05239305bdb1bd42f0a1f1f',
 'src/contiguous_fuxian/flexgen_pcache.py':'b86364086588327f9b9a06e608968aa16b2ce20c',
 'src/contiguous_fuxian/sharedkv_v21/runtime.py':'69e088baf05cc64009850a52a69792831123d0ca',
 'src/contiguous_fuxian/sharedkv_v21/ops.py':'e7b15b1407689b06673159d23774fd2c97e9ed64',
 'src/contiguous_fuxian/sharedkv/core.py':'d83ff10040329bea613e1cd5c5c7b95c1e1e6ff7',
 'src/contiguous_fuxian/sharedkv/io.py':'5cc0bc761955df308e3b352d62cf48c3660f3b05',
 'src/contiguous_fuxian/sharedkv/runtime.py':'944ee1cd079cd75cc4e1d7b44000b6f1c679ad2c',
}
def check(root):
 for name,expected in CHECKS.items():
  content=(Path(root)/name).read_bytes()
  actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
  if actual!=expected:raise RuntimeError(f'Interface mismatch {name}; expected {BASE}. Re-audit, do not disable this guard.')
