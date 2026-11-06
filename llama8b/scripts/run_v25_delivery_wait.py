from pathlib import Path
import time,json,subprocess,os
r=Path(__file__).resolve().parents[1]
while True:
 t=(r/'logs/v25_diagnostics.log').read_text()
 if 'Traceback (most recent call last)' in t:raise RuntimeError('v25 diagnostics failed')
 if 'ALL V25 DIAGNOSTICS COMPLETE' in t:break
 time.sleep(10)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',PYTHONPATH='src:runtime_deps_v24:runtime_deps_plot:runtime_deps')
py='/home/panzihang/venvs/vllm-stable/bin/python'
for script in ['check_v25_invariants.py','analyze_v25_results.py','plot_v25_results.py','write_v25_report.py','archive_v25.py']:
 print('START',script,flush=True)
 subprocess.run([py,'-u','scripts/'+script],cwd=r,env=env,check=True)
 print('DONE',script,flush=True)
print('ALL V25 DELIVERY COMPLETE',flush=True)
