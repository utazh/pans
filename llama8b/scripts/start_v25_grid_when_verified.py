
from pathlib import Path
import json,time,subprocess,os
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
while True:
 a=(r/'logs/v25_bridge.log').read_text();b=(r/'logs/v25_smoke_r4_online_parity.log').read_text()
 if 'Traceback (most recent call last)' in a+b:raise RuntimeError('Bridge/parity failed; investigate before training')
 p=r/'audit/v25_smoke_r4_online_parity.jsonl'
 if 'SUITE COMPLETE' in a and p.exists() and len(p.read_text().splitlines())==8:break
 time.sleep(5)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',PYTHONPATH='src:runtime_deps_v24',CUDA_VISIBLE_DEVICES='')
subprocess.run([py,'scripts/verify_v25_bridge.py'],cwd=r,env=env,check=True)
children=[]
for R,gpu in [(0,'2'),(4,'3')]:
 with (r/f'logs/v25_grid_r{R}.log').open('x') as f:p=subprocess.Popen([py,'-u','scripts/train_v25_grid_worker.py','--R',str(R),'--gpu',gpu],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 children.append(p);print('GRID PID',R,p.pid,flush=True)
codes=[p.wait() for p in children]
assert codes==[0,0],codes
print('ALL V25 GRID TRAINING COMPLETE',flush=True)
