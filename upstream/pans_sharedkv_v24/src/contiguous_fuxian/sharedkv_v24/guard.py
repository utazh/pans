"""Pinned published v2.3 interface verification. Never bypass on mismatch."""
from pathlib import Path
import hashlib
BASE='3868c6b64332f89cc0b301c75b53c35630302f0e'
CHECKS={
'src/contiguous_fuxian/sharedkv_v23/block.py':'dd9d0d0fe1ce53229558da3379009aa58678955a',
'src/contiguous_fuxian/sharedkv_v23/train.py':'922b45278596d66e56f47505836aa3000dbf3996',
'src/contiguous_fuxian/sharedkv_v23/capture.py':'84cd0da0f37c9aa6b6204c1de205ac65629a2c0a',
'scripts/run_sharedkv_v23.py':'c561df19113be0b97082445eea348e664836120c',
'src/contiguous_fuxian/sharedkv/io.py':'5cc0bc761955df308e3b352d62cf48c3660f3b05',
'src/contiguous_fuxian/sharedkv/fit.py':'4da7d62b41ef265631c311eea5a3f78d882fa478',
'src/contiguous_fuxian/flexgen_qwen_reprefill.py':'d52f1f610b7e63d9b150f7bfdba2caaa2866101e',
'src/contiguous_fuxian/flexgen_pcache.py':'b86364086588327f9b9a06e608968aa16b2ce20c',
'vendor/flexgen/my_pcache_fast.py':'9dae80e9d24ee53c2c57c9a86153d3ad1a72ff5d',
}
def check(root):
    for name,expected in CHECKS.items():
        b=(Path(root)/name).read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if actual!=expected:raise RuntimeError(f'Interface changed: {name}; expected {BASE}. Re-audit rather than bypass.')
