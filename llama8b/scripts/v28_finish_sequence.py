from pathlib import Path
import os,json,time,subprocess,hashlib
r=Path('/home/panzihang/src/prism_layerkv');os.chdir(r)
py='/home/panzihang/venvs/vllm-stable/bin/python'
env=dict(os.environ,PYTHONPATH='src:runtime_deps_v24',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
deps=[('main','audit/v28_main_process.json'),('factors','audit/v28_factors_process.json'),('external','audit/v28_external_process_attempt2.json')]
while True:
 waiting=[]
 for name,p in deps:
  if (r/f'audit/v28_{name}_done.json').exists():continue
  pid=json.loads((r/p).read_text())['pid']
  if not Path(f'/proc/{pid}').exists():raise RuntimeError(f'{name} stopped without completion; timing must not start')
  waiting.append(name)
 if not waiting:break
 time.sleep(30)
print('QUALITY_AND_PREPARATION_COMPLETE; START_SERIAL_TIMING',flush=True)
cmd=[py,'-u','scripts/matrix_v28.py','run','--manifest','studies/v28_frozen/manifest.json','--stage','timing','--execute']
start=time.time();rc=subprocess.call(cmd,env=env)
(r/'audit/v28_timing_exit.json').write_text(json.dumps(dict(argv=cmd,returncode=rc,seconds=time.time()-start)))
if rc:raise SystemExit(rc)
subprocess.run([py,'scripts/matrix_v28.py','analyze','--manifest','studies/v28_frozen/manifest.json','--output','results/v28_summary'],env=env,check=True)
subprocess.run([py,'scripts/v28_write_report.py'],env=env,check=True)
with (r/'logs/v28_final_regression.log').open('x') as f:
 subprocess.run([py,'-m','pytest','-q','tests'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
initial=json.loads((r/'audit/v28_initial_protection.json').read_text());changed={}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
for kind,files in initial.items():changed[kind]=[p for p,h in files.items() if not Path(p).is_file() or sha(p)!=h]
(r/'audit/v28_final_protection.json').write_text(json.dumps(dict(checked={k:len(v) for k,v in initial.items()},changed=changed,passed=not any(changed.values())),indent=2))
(r/'audit/v28_all_done.json').write_text(json.dumps(dict(completed=time.time(),report='EXPERIMENT_REPORT_V28_ZH.md')))
print('V28_ALL_DONE',flush=True)
