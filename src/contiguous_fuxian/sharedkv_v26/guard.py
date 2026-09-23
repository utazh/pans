"""Fail closed on changes to interfaces read through the GitHub connector."""
from pathlib import Path
import hashlib
BASE = '5e523ec7f48b99b54cf449ace7b8f95e94c7e8db'
CHECKS = {
    'src/contiguous_fuxian/sharedkv_v25/runtime.py': 'ba99cb18db1f20ce50b4388d61db036f5be7a3ad',
    'src/contiguous_fuxian/sharedkv_v25/attention.py': '1b7bd7262cd5ee7ab1df4f0d6b906254c08048bb',
    'src/contiguous_fuxian/sharedkv_v25/policy.py': '46faea919c0d92b8be3867ac0b4ae6a9cf5ea3b4',
    'scripts/run_native_prefix_v25.py': 'b1831f0ad774421ed919da8b538b879a51228fe2',
    'src/contiguous_fuxian/flexgen_pcache.py': 'b86364086588327f9b9a06e608968aa16b2ce20c',
    'vendor/flexgen/my_pcache_fast.py': '9dae80e9d24ee53c2c57c9a86153d3ad1a72ff5d',
}

def check(root: str | Path) -> None:
    for name, expected in CHECKS.items():
        content = (Path(root) / name).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Interface changed: {name}; re-audit instead of disabling the guard ({BASE}).')
