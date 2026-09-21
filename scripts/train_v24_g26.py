from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',PYTHONPATH=str(r/'src')+':'+str(r/'runtime_deps_v24'))
with (r/'logs/v24_repaired_tests.log').open('x') as f:
 subprocess.run([py,'-m','pytest','-q','tests/test_sharedkv_v24.py','tests/test_sharedkv_v24_server.py','-p','no:cacheprovider'],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
base=json.loads((r/'configs/v24_smoke_fa2_command.json').read_text())
for method in ['score','nomargin']:
 for seed in [24,25,26]:
  name=f'v24_{method}_g26_s{seed}';cmd=base.copy()
  for flag,value in [('--worklog',str(r/f'audit/{name}_compute.json')),('--steps','200'),('--eval-every','25'),('--seed',str(seed)),('--output',str(r/f'assets/{name}/trec.pt'))]:cmd[cmd.index(flag)+1]=value
  if method=='nomargin':cmd+=['--rank-weight','0']
  (r/f'configs/{name}_command.json').write_text(json.dumps(cmd,indent=2))
  print('START',name,flush=True)
  with (r/f'logs/{name}.log').open('x') as f:subprocess.run(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',name,flush=True)
print('G26 SCORE TRAINING COMPLETE',flush=True)
