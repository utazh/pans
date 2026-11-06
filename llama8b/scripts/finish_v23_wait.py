import json,os,time,subprocess
from pathlib import Path
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
while True:
 log=(r/'logs/v23_postprocess.log').read_text()
 if 'POSTPROCESSING COMPLETE' in log:break
 if 'Traceback (most recent call last)' in log:raise RuntimeError('Postprocessing failed')
 time.sleep(10)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',PYTHONPATH='src')
for cmd,log in [
 ([py,'-u','scripts/run_layerkv_v23_suite.py','configs/v23_decode_jobs.json'],'v23_decode.log'),
 ([py,'-m','pytest','-q','tests'],'v23_all_tests.log'),
 ([py,'scripts/check_v23_invariants.py'],'v23_invariants.log'),
 ([py,'scripts/plot_v23_results.py'],'v23_plot.log')]:
 print('START',cmd,flush=True)
 with (r/'logs'/log).open('w') as f:subprocess.run(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',cmd,flush=True)
print('VALIDATION AND PLOTS COMPLETE',flush=True)
