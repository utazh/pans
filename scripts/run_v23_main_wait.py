import os,time,subprocess,json
from pathlib import Path
r=Path('/home/panzihang/src/prism_layerkv')
def alive(pid):
 p=Path(f'/proc/{pid}/stat')
 return p.exists() and p.read_text().split()[2]!='Z'
while any(alive(p) for p in [1128,6565]):time.sleep(3)
assert (r/'results/v23_dev_query/summary.json').is_file()
assert (r/'assets/v23_query_full_validation/trec.pt').is_file()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
print('MAIN FULL500 WARM/COLD COMPARISON START',flush=True)
subprocess.run(['/home/panzihang/venvs/vllm-stable/bin/python','-u','scripts/run_layerkv_v23_suite.py','configs/v23_main_jobs.json'],cwd=r,env=env,check=True)
print('MAIN EXPERIMENTS COMPLETE',flush=True)
