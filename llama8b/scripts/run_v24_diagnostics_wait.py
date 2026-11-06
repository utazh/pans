"""Run v24 diagnostics only after all formal timing has finished."""
from pathlib import Path
import os,json,time,subprocess
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(r/'src')+':'+str(r/'runtime_deps_v24')+':'+str(r/'runtime_deps_plot')+':'+str(r/'runtime_deps'))
while True:
 source=r/'logs/v24_cold_resume.log'
 if not source.exists():source=r/'logs/v24_ladder_eval.log'
 txt=source.read_text()
 if 'Traceback (most recent call last)' in txt:raise RuntimeError('Main experiment failed')
 if 'LADDER QUALITY TIMING COMPLETE' in txt or 'COLD FORMAL COMPLETE' in txt:break
 time.sleep(10)
def call(cmd,name):
 (r/f'configs/{name}_command.json').write_text(json.dumps(cmd,indent=2))
 print('START',name,flush=True)
 with (r/f'logs/{name}.log').open('x') as f:subprocess.run(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,flush=True)
call([py,'-m','pytest','-q','tests','-p','no:cacheprovider'],'v24_all_gpu_tests')
jobs=json.loads((r/'configs/v24_formal_timing_jobs.json').read_text())
def args(policy,arm,name,n=8,warm=4):
 a=next(x for x in jobs if x[x.index('--output')+1].endswith(f'v24_time_{policy}_b0_{arm}')).copy()
 old=a[a.index('--output')+1].split('/')[-1];a=[x.replace(old,name) for x in a]
 a[a.index('--samples')+1]=str(n);a[a.index('--warmup')+1]=str(warm)
 a.insert(a.index('--')+1,'--diagnostic')
 return a
# Post hoc lifecycle confirmation, declared before observing these new timings.
for block in range(4):
 order=['identity','shared'] if block in (0,3) else ['shared','identity']
 for arm in order:
  name=f'v24_confirm_warm_b{block}_{arm}'
  a=args('warm',arm,name,n=32,warm=8);a.remove('--diagnostic')
  job=r/f'configs/{name}_job.json'
  with job.open('x') as f:json.dump([a],f,indent=2)
  call([py,'-u','scripts/run_layerkv_v24_suite.py',str(job)],name)
for arm in ['identity','shared']:
 name='v24_private_'+arm
 a=args('warm',arm,name,warm=0);a[a.index('--max-tokens')+1]='16'
 call([py,'-u','scripts/audit_v24_decode.py','--audit-output',str(r/f'audit/{name}.jsonl'),'--',*a],name)
name='v24_shared_replay_parity'
call([py,'-u','scripts/audit_v24_shared_replay.py','--audit-output',str(r/f'audit/{name}.jsonl'),'--',*args('warm','shared',name,warm=0)],name)
for policy in ['native','preserve']:
 for arm in ['identity','shared']:
  name=f'v24_restore_{policy}_{arm}';a=args(policy,arm,name)
  call([py,'-u','scripts/run_sharedkv_v24.py','--audit-restore',str(r/f'audit/{name}.jsonl'),'--',*a],name)
nsys=str(r/'runtime_tools/nsight-2025.5.1/opt/nvidia/nsight-systems-cli/2025.5.1/target-linux-x64/nsys')
(r/'profiles/v24').mkdir(exist_ok=True)
for policy in ['native','preserve','warm']:
 for arm in ['identity','shared']:
  name=f'v24_profile_nsys_{policy}_{arm}';a=args(policy,arm,name);a.insert(a.index('--')+1,'--nvtx')
  stem=r/f'profiles/v24/nsys_{policy}_{arm}'
  cmd=[nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output',str(stem),py,'-u','scripts/profile_layerkv_v24.py','--profiler','nsys','--profile-output',str(stem),'--',*a]
  call(cmd,name)
  subprocess.run([nsys,'export','--type=sqlite','--output',str(stem)+'.sqlite',str(stem)+'.nsys-rep'],env=env,cwd=r,check=True)
for policy,arm in [(p,a) for p in ['native','warm'] for a in ['identity','shared']]:
 name=f'v24_profile_viz_{policy}_{arm}';a=args(policy,arm,name);a.insert(a.index('--')+1,'--nvtx')
 call([py,'-u','scripts/profile_layerkv_v24.py','--profiler','viztracer','--profile-output',str(r/f'profiles/v24/viztracer_{policy}_{arm}'),'--',*a],name)
call([py,'scripts/analyze_v24_nsight.py'],'v24_nsight_analysis')
call([py,'scripts/analyze_v24_viz.py'],'v24_viz_analysis')
print('ALL V24 DIAGNOSTICS COMPLETE',flush=True)
