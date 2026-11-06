import os,time,subprocess
from pathlib import Path
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
while 'SUITE COMPLETE' not in (r/'logs/v23_cold_probe8.log').read_text():
 if 'Traceback (most recent call last)' in (r/'logs/v23_cold_probe8.log').read_text():raise RuntimeError('Cold probe failed')
 time.sleep(2)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4')
print('ISOLATED COLD FULL500 PHASE START',flush=True)
subprocess.run([py,'-u','scripts/run_layerkv_v23_suite.py','configs/v23_cold_resume_jobs.json'],cwd=r,env=env,check=True)
print('COLD PHASE COMPLETE; FRESH PROCESS WARM TIMING CONFIRMATION',flush=True)
subprocess.run([py,'-u','scripts/run_layerkv_v23_suite.py','configs/v23_time_confirm_jobs.json'],cwd=r,env=env,check=True)
print('MAIN EXPERIMENTS COMPLETE',flush=True)
