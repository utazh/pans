"""Run each cold arm in its own process; unchanged warmup/cold/TTFT logic."""
from pathlib import Path
import os,sys,json,subprocess
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',PYTHONPATH=str(r/'src')+':'+str(r/'runtime_deps_v24'))
for a in json.loads(Path(sys.argv[1]).read_text()):
 assert '--cold' in a and a[a.index('--cold')+1]=='ssd-cold'
 name=Path(a[a.index('--output')+1]).name
 job=r/f'configs/{name}_isolated_job.json'
 with job.open('x') as f:json.dump([a],f,indent=2)
 print('START ISOLATED COLD',name,flush=True)
 with (r/f'logs/{name}.log').open('x') as f:
  subprocess.run([py,'-u','scripts/run_layerkv_v24_suite.py',str(job)],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE ISOLATED COLD',name,flush=True)
print('COLD FORMAL COMPLETE',flush=True)
