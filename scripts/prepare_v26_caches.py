from pathlib import Path
import shutil,os,json,time
r=Path('/home/panzihang/src/prism_layerkv'); src=r/'runtime_data/pcache_c16';report={}
for gpu in [3,2]:
 dst=r/f'runtime_data/v26_pcache_gpu{gpu}'
 assert not dst.exists()
 shutil.copytree(src,dst,copy_function=shutil.copyfile)
 n=sz=0
 for p in dst.rglob('*'):
  if not p.is_file():continue
  a=p.stat();b=(src/p.relative_to(dst)).stat()
  assert a.st_nlink==1 and (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino)
  with p.open('rb') as f:os.fsync(f.fileno())
  n+=1;sz+=a.st_size
 report[str(gpu)]={'root':str(dst),'files':n,'bytes':sz,'independent_inodes':True}
 print('CACHE READY',gpu,n,sz,flush=True)
with (r/'audit/v26_cache_copies.json').open('x') as f:json.dump(report,f,indent=2)
