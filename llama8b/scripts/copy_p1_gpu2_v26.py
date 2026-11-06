from v26_common import *
import shutil
a=ROOT/'runtime_data/v26_p1_pcache_gpu3';b=ROOT/'runtime_data/v26_p1_pcache_gpu2'
shutil.copytree(a,b)
os.sync()
with (ROOT/'audit/v26_p1_gpu2_copy_done.json').open('x') as f:json.dump(dict(source=str(a),target=str(b),files=sum(p.is_file() for p in b.rglob('*')),independent_copy=True),f)
print('COPY DONE')
