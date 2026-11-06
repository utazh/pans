"""Exact known integration surfaces of the published v2.2 branch."""
from pathlib import Path
import hashlib
BASE='fddabb7015c66c870a11810069bfbe20d5a34bbf'
CHECKS={
'src/contiguous_fuxian/sharedkv_v21/runtime.py':'69e088baf05cc64009850a52a69792831123d0ca',
'src/contiguous_fuxian/flexgen_qwen_reprefill.py':'d52f1f610b7e63d9b150f7bfdba2caaa2866101e',
'src/contiguous_fuxian/flexgen_pcache.py':'b86364086588327f9b9a06e608968aa16b2ce20c',
'vendor/flexgen/my_pcache_fast.py':'9dae80e9d24ee53c2c57c9a86153d3ad1a72ff5d',
'src/contiguous_fuxian/priority_executor.py':'8b2f2d1531502bb1812302e3df67d382de256e52',
'src/contiguous_fuxian/sharedkv/io.py':'5cc0bc761955df308e3b352d62cf48c3660f3b05',
'src/contiguous_fuxian/sharedkv/fit.py':'4da7d62b41ef265631c311eea5a3f78d882fa478',
}

def check(root):
    for path,expected in CHECKS.items():
        data=(Path(root)/path).read_bytes()
        got=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if got!=expected:raise RuntimeError(f'Changed interface {path}: re-audit against {BASE}, do not bypass')
