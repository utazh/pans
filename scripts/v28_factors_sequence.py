from pathlib import Path
import os,json,time,torch
from contiguous_fuxian.sharedkv_v28.factors import compile_cell
from contiguous_fuxian.sharedkv_v28.common import sha,write_json,assert_hashes
r=Path('/home/panzihang/src/prism_layerkv');os.chdir(r);torch.set_num_threads(4)
m=json.loads((r/'studies/v28_frozen/manifest.json').read_text());assert_hashes(m['input_sha256'])
cells=sorted(m['cells'],key=lambda c:(c['id']!='trec_b25',c['task'],c['budget']))
for c in cells:
 start=time.time();print('COMPILE_FACTORS',c['id'],flush=True)
 compile_cell(c,m['spec'],m['root'])
 out=Path(m['root'])/'representations'/c['id']
 write_json(r/f"audit/v28_factors_{c['id']}.json",dict(seconds=time.time()-start,files={str(p):sha(p) for p in out.glob('*.pt')},disk_free=__import__('shutil').disk_usage(r).free))
write_json(r/'audit/v28_factors_done.json',dict(completed=time.time()))
