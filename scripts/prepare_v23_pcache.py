from pathlib import Path
import shutil,os,json,time
root=Path('/home/panzihang/src/prism_layerkv');src=root/'runtime_data/pcache_c16';dst=root/'runtime_data/v23_pcache_ssd'
assert not dst.exists();shutil.copytree(src,dst,copy_function=shutil.copyfile)
count=total=0
for p in dst.rglob('*'):
 if not p.is_file():continue
 s=p.stat();orig=(src/p.relative_to(dst)).stat()
 assert s.st_nlink==1 and (s.st_dev,s.st_ino)!=(orig.st_dev,orig.st_ino)
 with p.open('rb') as f:os.fsync(f.fileno())
 count+=1;total+=s.st_size
(root/'audit/v23_pcache_copy.json').write_text(json.dumps({'source':str(src),'destination':str(dst),'files':count,'bytes':total,'independent_inodes':True,'fsync_completed':True,'time':time.time()},indent=2))
print('COPY COMPLETE',count,total,flush=True)
