
from pathlib import Path
import shutil,json,os,time
r=Path('/home/panzihang/src/prism_layerkv');src=r/'runtime_data/v23_pcache_ssd';report={}
for gpu in [2,3]:
 dst=r/f'runtime_data/v24_pcache_gpu{gpu}';assert not dst.exists()
 shutil.copytree(src,dst,copy_function=shutil.copyfile)
 files=0;size=0
 for p in dst.rglob('*'):
  if not p.is_file():continue
  a=p.stat();b=(src/p.relative_to(dst)).stat()
  assert a.st_nlink==1 and a.st_size==b.st_size and (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino)
  with p.open('rb') as f:os.fsync(f.fileno())
  files+=1;size+=a.st_size
 report[str(gpu)]={'root':str(dst),'files':files,'bytes':size,'independent_inodes':True,'fsync_complete':True}
(r/'audit/v24_pcache_copies.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
