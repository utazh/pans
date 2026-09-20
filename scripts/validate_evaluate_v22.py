import subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parents[1]
with (r/'logs/v22_all_tests.log').open('w') as f:
 subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True)
for t in (19,27):
 subprocess.run([sys.executable,'scripts/bench_headalign_v22.py','--adapter',f'assets/v22_map{t}_s200/trec.pt','--target',str(t),'--prefix-tokens','1228','--query-tokens','41','--iterations','200','--warmup','30','--output',f'results/v22_micro_map{t}_q41.json'],cwd=r,check=True)
subprocess.run([sys.executable,'scripts/budget_frontier_v22.py','--output','results/v22_budget_arithmetic.json'],cwd=r,check=True)
subprocess.run([sys.executable,'-u','scripts/run_layerkv_v22_suite.py','scripts/v22_quality_s200_jobs.json'],cwd=r,check=True)
print('VALIDATION AND QUALITY COMPLETE',flush=True)
