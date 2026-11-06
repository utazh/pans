from pathlib import Path
import json,subprocess,os,time
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
while True:
 txt=(r/'logs/v24_diagnostics.log').read_text()
 if 'Traceback (most recent call last)' in txt:raise RuntimeError('Diagnostics failed')
 if 'ALL V24 DIAGNOSTICS COMPLETE' in txt:break
 time.sleep(10)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',PYTHONPATH=str(r/'src')+':'+str(r/'runtime_deps_v24')+':'+str(r/'runtime_deps_plot'))
for script in ['check_v24_invariants.py','analyze_v24_results.py','plot_v24_results.py','write_v24_report.py','archive_v24.py']:
 print('RUN',script,flush=True)
 subprocess.run([py,'scripts/'+script],cwd=r,env=env,check=True)
print('V24 DELIVERY COMPLETE',flush=True)
