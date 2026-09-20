import subprocess,sys,time,json
from pathlib import Path
r=Path(__file__).resolve().parents[1]
def wait_pid(pid):
 start=time.time()
 while Path(f'/proc/{pid}').exists():
  if time.time()-start>1800:raise TimeoutError(f'Prior process {pid} not finished')
  time.sleep(3)
def run(args):
 print('STAGE',args,flush=True);subprocess.run([sys.executable,'-u',*args],cwd=r,check=True)
wait_pid(27756)
assert 'VALIDATION AND QUALITY COMPLETE' in (r/'logs/v22_quality_s200.log').read_text()
run(['scripts/run_layerkv_v22_suite.py','scripts/v22_controls_jobs.json'])
wait_pid(24749)
assert 'SUITE COMPLETE' in (r/'logs/v22_quality_s0.log').read_text()
assert len(list((r/'results').glob('v22_reg_*/summary.json')))==9
run(['scripts/run_layerkv_v22_suite.py','scripts/v22_timing_jobs.json'])
run(['scripts/profile_suite_v22.py'])
run(['scripts/analyze_layerkv_v21.py','--prefix','v22_reg_','--baseline','v22_reg_identity','--output','results/v22_quality_compare.json'])
run(['scripts/analyze_layerkv_v21.py','--prefix','v22_fixed_','--baseline','v22_fixed_identity','--output','results/v22_fixed_compare.json'])
run(['scripts/check_v22_invariants.py'])
print('V22 PIPELINE COMPLETE',flush=True)
