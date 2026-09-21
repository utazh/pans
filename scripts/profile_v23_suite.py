#!/usr/bin/env python3
import json,os,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
nsys=str(r/'runtime_tools/nsight-2025.5.1/opt/nvidia/nsight-systems-cli/2025.5.1/target-linux-x64/nsys')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
main=json.loads((r/'configs/v23_main_jobs.json').read_text())
def args(arm,cold,name):
 src=next(a for a in main if a[a.index('--output')+1].endswith('v23_cold_b00_'+arm if cold else 'v23_full_'+arm))
 old=src[src.index('--output')+1].split('/')[-1]
 a=[x.replace(old,name) for x in src]
 a[a.index('--samples')+1]='8';a[a.index('--warmup')+1]='4'
 if '--nvtx' not in a:
  sep=a.index('--');a.insert(sep+1,'--nvtx')
 if '--diagnostic' not in a:
  a.insert(a.index('--')+1,'--diagnostic')
 return a
(r/'profiles/v23').mkdir(exist_ok=True)
for name,profiler,arm,cold in [('cold_identity','nsys','identity',True),('cold_joint','nsys','joint',True),('cold_joint','viztracer','joint',True),('warm_joint','viztracer','joint',False)]:
 out=r/f'profiles/v23/{profiler}_{name}';extra=args(arm,cold,f'v23_profile_{profiler}_{name}')
 cmd=[py,'-u','scripts/profile_layerkv_v23.py','--profiler',profiler,'--profile-output',str(out),'--',*extra]
 if profiler=='nsys':
  stem=r/f'profiles/v23/nsys_{name}'
  cmd=[nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output',str(stem),*cmd]
 print('PROFILE START',profiler,name,flush=True)
 with (r/f'logs/v23_profile_{profiler}_{name}.log').open('w') as f:
  subprocess.run(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 if profiler=='nsys':
  subprocess.run([nsys,'export','--type=sqlite','--output',str(stem)+'.sqlite',str(stem)+'.nsys-rep'],env=env,cwd=r,check=True)
 print('PROFILE DONE',profiler,name,flush=True)
print('PROFILES COMPLETE',flush=True)
