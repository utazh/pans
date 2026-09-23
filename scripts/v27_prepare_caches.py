
from pathlib import Path
import shutil,os,json,hashlib,subprocess
r=Path('/home/panzihang/src/prism_layerkv');os.chdir(r)
src=r/'runtime_data/pcache_c16'
for gpu in [2,3]:
 dst=r/f'runtime_data/v27_pcache_gpu{gpu}';shutil.copytree(src,dst)
 assert len(list(src.iterdir()))==len(list(dst.iterdir()))
 for p in src.iterdir():
  q=dst/p.name
  if p.is_file():
   assert p.stat().st_ino!=q.stat().st_ino and p.read_bytes()==q.read_bytes()
 print('verified cache',gpu,flush=True)
os.sync()
env=dict(os.environ,PYTHONPATH='src:runtime_deps_v24',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
code=subprocess.call(['/home/panzihang/venvs/vllm-stable/bin/python','-u','scripts/v27_stage_a2.py'],env=env)
(r/'audit/v27_stage_a2_exit.json').write_text(json.dumps({'returncode':code}))
