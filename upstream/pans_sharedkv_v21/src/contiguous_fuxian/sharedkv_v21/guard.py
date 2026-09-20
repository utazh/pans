"""Git blob identities of the published v1 interfaces, not the old main branch."""
import hashlib
from pathlib import Path
BASE_V1='59312f2bd2504079fed03b53f283d8ade931fee2'
CHECKS={
'scripts/run_round4.py':'d9b973eb528c26e713f75b8620ec7cc9608b8dc8',
'src/contiguous_fuxian/flexgen_qwen_reprefill.py':'d52f1f610b7e63d9b150f7bfdba2caaa2866101e',
'src/contiguous_fuxian/flexgen_pcache.py':'b86364086588327f9b9a06e608968aa16b2ce20c',
'src/contiguous_fuxian/sharedkv/runtime.py':'944ee1cd079cd75cc4e1d7b44000b6f1c679ad2c',
'src/contiguous_fuxian/sharedkv/core.py':'d83ff10040329bea613e1cd5c5c7b95c1e1e6ff7',
'src/contiguous_fuxian/sharedkv/io.py':'5cc0bc761955df308e3b352d62cf48c3660f3b05',
'src/contiguous_fuxian/sharedkv/fit.py':'4da7d62b41ef265631c311eea5a3f78d882fa478',
'src/contiguous_fuxian/sharedkv/selection.py':'928c12cb598920e83140f8b80f204bc514cd3f3d'}
def check(repo):
    for path,expected in CHECKS.items():
        b=(Path(repo)/path).read_bytes()
        h=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if h!=expected:raise RuntimeError(f'Interface changed: {path}; expected published v1 {BASE_V1}. Re-audit, do not bypass.')
