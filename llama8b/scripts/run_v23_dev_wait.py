import os,time,subprocess
from pathlib import Path
r=Path('/home/panzihang/src/prism_layerkv')
while any(Path('/proc/'+str(pid)).exists() for pid in (55356,59660)):
 time.sleep(3)
assert all((r/f'assets/v23_{a}/trec.pt').is_file() for a in ('joint','independent','query'))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
subprocess.run(['/home/panzihang/venvs/vllm-stable/bin/python','-u','scripts/run_layerkv_v23_suite.py','configs/v23_dev_jobs.json'],cwd=r,env=env,check=True)
