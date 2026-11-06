from pathlib import Path
import subprocess,os,json,time,importlib.util
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='4',PYTHONPATH=str(r/'src')+':'+str(r/'runtime_deps_v24'))
while 'GROUP DEVELOPMENT COMPLETE' not in (r/'logs/v24_group_dev.log').read_text():
 if 'Traceback (most recent call last)' in (r/'logs/v24_group_dev.log').read_text():raise RuntimeError('Group dev failed')
 time.sleep(10)
base=json.loads((r/'configs/v24_group_dev_jobs.json').read_text())[1]
for method in ['score','nomargin']:
 for seed in [24,25,26]:
  name=f'v24_{method}_g26_s{seed}';work=r/f'audit/{name}_compute.json'
  while not work.exists():
   if 'Traceback (most recent call last)' in (r/'logs/v24_g26_training.log').read_text():raise RuntimeError('Training failed')
   time.sleep(10)
  if json.loads(work.read_text())['status']!='completed':raise RuntimeError(name+' training failed')
  args=base.copy()
  for flag,value in [('--stats',str(r/f'results/v24_dev_{method}_g26_s{seed}.extra.jsonl')),('--adapters',str(r/f'assets/{name}')),('--output',str(r/f'results/v24_dev_{method}_g26_s{seed}'))]:args[args.index(flag)+1]=value
  jobs=r/f'configs/{name}_dev_job.json';jobs.write_text(json.dumps([args],indent=2))
  print('START QUALITY',name,flush=True)
  subprocess.run([py,'-u','scripts/run_layerkv_v24_suite.py',str(jobs)],cwd=r,env=env,check=True)
  subprocess.run([py,'scripts/audit_quality_v24.py','--baseline',str(r/'results/v24_dev_identity/scored_records.jsonl'),'--student',str(r/f'results/v24_dev_{method}_g26_s{seed}/scored_records.jsonl'),'--output',str(r/f'audit/{name}_dev_quality.json')],cwd=r,env=env,check=True)
  print('DONE QUALITY',name,flush=True)
print('G26 DEVELOPMENT COMPLETE',flush=True)
