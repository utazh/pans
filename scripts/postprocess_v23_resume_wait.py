import json,os,time,subprocess
from pathlib import Path
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
# Wait for completed suite sentinel, abort if main suite exits with an error.
while True:
 log=(r/'logs/v23_main_resume.log').read_text()
 if 'MAIN EXPERIMENTS COMPLETE' in log:break
 if 'Traceback (most recent call last)' in log:raise RuntimeError('Main experiment failed; inspect v23_main.log')
 time.sleep(10)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',PYTHONPATH='src')
with (r/'logs/v23_analysis.log').open('w') as f:
 subprocess.run([py,'scripts/analyze_v23_results.py'],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run([py,'-u','scripts/profile_v23_suite.py'],cwd=r,env=env,check=True)
for script in ('analyze_v23_nsight.py','analyze_v23_viz_cold.py','analyze_v23_viz_warm.py'):
 subprocess.run([py,'scripts/'+script],cwd=r,env=env,check=True)
print('POSTPROCESSING COMPLETE',flush=True)
