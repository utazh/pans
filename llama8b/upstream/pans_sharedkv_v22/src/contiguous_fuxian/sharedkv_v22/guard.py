"""Pinned v2 source identities obtained from the GitHub connector on review."""
import hashlib
from pathlib import Path
from .matching import BASE_V2
EXTRA_CHECKS={
 'scripts/run_sharedkv_v21.py':'2687662615672bece81e18b8dd957a8fc69f74ef',
 'src/contiguous_fuxian/sharedkv_v21/runtime.py':'69e088baf05cc64009850a52a69792831123d0ca',
 'src/contiguous_fuxian/sharedkv_v21/refine.py':'4093410f35cb0c1fd618c0d64eb0a7aad134bd7f',
 'src/contiguous_fuxian/sharedkv_v21/ops.py':'e7b15b1407689b06673159d23774fd2c97e9ed64',
}
def check(repo):
    from contiguous_fuxian.sharedkv_v21.guard import check as original_check
    original_check(repo)
    for path,expected in EXTRA_CHECKS.items():
        b=(Path(repo)/path).read_bytes()
        h=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if h!=expected:raise RuntimeError(f'Interface changed: {path}; re-audit against v2 {BASE_V2}, do not bypass.')
