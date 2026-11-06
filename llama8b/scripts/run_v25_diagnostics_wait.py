
import json,time,subprocess
from v25_common import *
while True:
 t=(ROOT/'logs/v25_pipeline.log').read_text()
 if 'Traceback (most recent call last)' in t:raise RuntimeError('Main v25 experiment failed')
 if 'ALL V25 QUALITY AND TIMING COMPLETE' in t:break
 time.sleep(10)
G=json.loads((ROOT/'audit/v25_frozen_selection.json').read_text())['chosen_G']
run('v25_all_gpu_tests',[PY,'-m','pytest','-q','tests','-p','no:cacheprovider'],3)
for arm,R in [('identity',None),('r0',0),('r4',4)]:
 name=f'v25_contract_{arm}'
 a=args(name,G=None if R is None else G,R=R,n=8,warm=0,gpu=3,diagnostic=True,max_tokens=16)
 run(name,[PY,'-u','scripts/audit_v25_contracts.py','--audit-output',str(ROOT/f'audit/{name}.jsonl'),'--',*a],3)
for R in [0,4]:
 name=f'v25_final_parity_r{R}';a=args(name,G=G,R=R,n=8,warm=0,gpu=3,diagnostic=True)
 run(name,[PY,'-u','scripts/audit_v25_shared_replay.py','--audit-output',str(ROOT/f'audit/{name}.jsonl'),'--',*a],3)
for policy in ['native','preserve']:
 for arm,R in [('identity',None),('r0',0),('shared',4)]:
  name=f'v25_restore_{policy}_{arm}';a=args(name,G=None if R is None else G,R=R,n=8,warm=4,cold=policy,diagnostic=True)
  run(name,[PY,'-u','scripts/audit_v25_restore.py','--audit-output',str(ROOT/f'audit/{name}.jsonl'),'--',*a],3)
profenv=env(3);profenv['PYTHONPATH']='src:runtime_deps_v24:runtime_deps_plot:runtime_deps'
nsys=str(ROOT/'runtime_tools/nsight-2025.5.1/opt/nvidia/nsight-systems-cli/2025.5.1/target-linux-x64/nsys')
(ROOT/'profiles/v25').mkdir(exist_ok=False)
def profile_call(name,cmd):
 with (ROOT/f'configs/{name}_command.json').open('x') as f:json.dump(cmd,f,indent=2)
 print('START',name,flush=True)
 with (ROOT/f'logs/{name}.log').open('x') as f:subprocess.run(cmd,cwd=ROOT,env=profenv,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,flush=True)
for policy in ['native','preserve','warm']:
 for arm,R in [('identity',None),('shared',4)]:
  name=f'v25_profile_nsys_{policy}_{arm}';a=args(name,G=None if R is None else G,R=R,n=8,warm=4,cold=None if policy=='warm' else policy,diagnostic=True);a.insert(a.index('--mode'),'--nvtx')
  stem=ROOT/f'profiles/v25/nsys_{policy}_{arm}'
  cmd=[nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output',str(stem),PY,'-u','scripts/profile_layerkv_v25.py','--profiler','nsys','--profile-output',str(stem),'--',*a]
  profile_call(name,cmd)
  subprocess.run([nsys,'export','--type=sqlite','--output',str(stem)+'.sqlite',str(stem)+'.nsys-rep'],cwd=ROOT,env=profenv,check=True)
for policy in ['native','warm']:
 for arm,R in [('identity',None),('shared',4)]:
  name=f'v25_profile_viz_{policy}_{arm}';a=args(name,G=None if R is None else G,R=R,n=8,warm=4,cold=None if policy=='warm' else policy,diagnostic=True);a.insert(a.index('--mode'),'--nvtx')
  profile_call(name,[PY,'-u','scripts/profile_layerkv_v25.py','--profiler','viztracer','--profile-output',str(ROOT/f'profiles/v25/viztracer_{policy}_{arm}'),'--',*a])
profile_call('v25_nsight_analysis',[PY,'scripts/analyze_v25_nsight.py'])
profile_call('v25_viz_analysis',[PY,'scripts/analyze_v25_viz.py'])
print('ALL V25 DIAGNOSTICS COMPLETE',flush=True)
