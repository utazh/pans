
from v26_common import *
# Existing primary calibration remains the owner of GPU2 until all jobs complete.
expected=[(t,b) for t in ['sst2','subj','rte','trec'] for b in [5,10,25,50] if (t,b)!=('trec',25)]
while not all((ROOT/f'audit/v26_p0_{t}_b{b}_train_process.json').exists() for t,b in expected):
 if 'Traceback' in (ROOT/'logs/v26_training_worker.log').read_text():raise RuntimeError('Primary calibration failed; not proceeding')
 time.sleep(10)
for t,b in expected:assert json.loads((ROOT/f'audit/v26_p0_{t}_b{b}_train_process.json').read_text())['returncode']==0
while not (ROOT/'audit/v26_admission_gpu_parity.json').exists():time.sleep(10)
run('v26_quality_gpu2',[PY,'-u','scripts/v26_quality_worker.py','--gpu','2'],2)
