
from pathlib import Path
import subprocess,os,time
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
while True:
 log=(r/'logs/v24_group_controls.log').read_text()
 if 'Traceback (most recent call last)' in log:raise RuntimeError('Group control training failed')
 if 'GROUP CONTROLS COMPLETE' in log and (r/'audit/v24_pcache_copies.json').exists():break
 time.sleep(10)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='4',PYTHONPATH=str(r/'src')+':'+str(r/'runtime_deps_v24'))
subprocess.run([py,'-u','scripts/run_layerkv_v24_suite.py','configs/v24_group_dev_jobs.json'],cwd=r,env=env,check=True)
print('GROUP DEVELOPMENT COMPLETE',flush=True)
