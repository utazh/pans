
from v26_common import *
import ast
before=json.loads((ROOT/'audit/v26_before.json').read_text());changed=[]
for path,sha in before['files'].items():
 p=Path(path)
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:changed.append(path)
assert not changed,changed
archive=ROOT/'artifacts/pans_sharedkv_v25_server_impl.zip'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==before['v25_archive_sha256']
for p in (ROOT/'scripts').glob('*v26*.py'):ast.parse(p.read_text(),filename=str(p))
assert json.loads((ROOT/'audit/v26_admission_gpu_parity.json').read_text())['passed']
# Verify both independent p0 physical cache copies still equal the untouched source.
hashes={};total=0
src=ROOT/'runtime_data/pcache_c16'
for p in sorted(src.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(src);sha=hashlib.sha256(p.read_bytes()).hexdigest();total+=p.stat().st_size
 for gpu in [2,3]:
  q=ROOT/f'runtime_data/v26_pcache_gpu{gpu}'/rel
  assert q.stat().st_ino!=p.stat().st_ino and q.stat().st_nlink==1 and hashlib.sha256(q.read_bytes()).hexdigest()==sha,(gpu,str(rel))
 hashes[str(rel)]=sha
with (ROOT/'audit/v26_cache_file_sha256.json').open('x') as f:json.dump(hashes,f,indent=2)

# Full p1 value audit, including every K/V payload chunk and selector-head file.
import numpy as np
p1_files=0
for pid,task in enumerate(['sst2','subj','trec','rte']):
 meta=json.loads((ROOT/f'runtime_data/v26_prefix_p1/{task}/metadata.json').read_text())
 n=meta['prefix_tokens'];chunks=(n+15)//16;shape=(n,meta['kv_heads'],meta['head_dim'])
 for lid in range(meta['layers']):
  for kind,short in [('key','k'),('value','v')]:
   raw=np.fromfile(ROOT/f'runtime_data/v26_prefix_p1/{task}/layer_{lid:02d}_{kind}.bf16',dtype=np.uint16)
   expected=(raw.astype(np.uint32)<<16).view(np.float32).reshape(shape).astype(np.float16)
   for cid in range(chunks):
    pos=cid if short=='k' else cid+chunks
    path=ROOT/f'runtime_data/v26_p1_pcache_gpu3/{pid}_{lid}_{pos}_{short}.npy'
    actual=np.fromfile(path,dtype=np.uint16)
    assert np.array_equal(actual,expected[cid*16:min(n,(cid+1)*16)].view(np.uint16).reshape(-1)),str(path)
    p1_files+=1
   if short=='k':
    path=ROOT/f'runtime_data/v26_p1_pcache_gpu3/{pid}_{lid}_None_k.npy'
    assert np.array_equal(np.fromfile(path,dtype=np.uint16),expected.view(np.uint16).reshape(-1)),str(path)
    p1_files+=1

for p in (ROOT/'runtime_data/v26_p1_pcache_gpu3').rglob('*'):
 if not p.is_file():continue
 q=ROOT/'runtime_data/v26_p1_pcache_gpu2'/p.relative_to(ROOT/'runtime_data/v26_p1_pcache_gpu3')
 assert q.stat().st_ino!=p.stat().st_ino and q.stat().st_nlink==1 and hashlib.sha256(q.read_bytes()).digest()==hashlib.sha256(p.read_bytes()).digest(),str(q)

result=dict(p1_payload_files_equal_prepared_prefix=p1_files,protected_files_unchanged=len(before['files']),v25_archive_unchanged=True,independent_p0_cache_copies_verified=2,files_per_copy=len(hashes),payload_bytes_per_copy=total,admission_parity_passed=True)
with (ROOT/'audit/v26_final_invariants.json').open('x') as f:json.dump(result,f,indent=2)
print(result)
