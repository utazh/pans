
from v26_common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
cells=([('p1',t,25) for t in ['sst2','subj','trec','rte']]+[('p0',t,b) for t in ['trec','sst2'] for b in [5,10,25,50]]) if a.gpu==3 else [('p0',t,b) for t in ['subj','rte'] for b in [5,10,25,50]]
assert json.loads((ROOT/'audit/v26_admission_gpu_parity.json').read_text())['passed']
if a.gpu==2:
 assert (ROOT/'audit/v26_subj_canonical_correction.json').exists()
 assert (ROOT/'audit/v26_p1_gpu2_copy_done.json').exists()
 while not (ROOT/'audit/v26_subj_canonical_training_done.json').exists():
  if 'Traceback' in (ROOT/'logs/v26_subj_canonical_sequence.log').read_text():raise RuntimeError('Corrected SUBJ calibration failed')
  time.sleep(10)
 cells=[('p1','subj',25)]+cells

for pref,task,budget in cells:
 tag=f'v26_{pref}_{task}_b{budget}'+('_canonical' if task=='subj' else '')
 adapter=ROOT/'assets/v25_g16_r4_s24' if (pref,task,budget)==('p0','trec',25) else ROOT/f'assets/{tag}'
 def checkpoint_complete():
  if (pref,task,budget)==('p0','trec',25):return True
  status=ROOT/f'audit/{tag}_train_process.json'
  try:state=json.loads(status.read_text())
  except (FileNotFoundError,json.JSONDecodeError):return False
  if state['returncode']!=0:raise RuntimeError('Calibration failed: '+tag)
  return (adapter/f'{task}.pt').is_file()
 while not checkpoint_complete():
  if 'Traceback' in (ROOT/'logs/v26_training_worker.log').read_text():raise RuntimeError('Primary calibration worker failed')
  time.sleep(10)
 sha=hashlib.sha256((adapter/f'{task}.pt').read_bytes()).hexdigest()
 with (ROOT/f'audit/{tag}_frozen.json').open('x') as f:json.dump(dict(checkpoint=str(adapter/f'{task}.pt'),sha256=sha,selection='fixed seed/graph/budget; validation checkpoint only'),f)
 for shared in [False,True]:
  name=tag+'_quality_'+('S' if shared else 'I')
  cmd=argv(name,task=task,budget=budget,gpu=a.gpu,shared=shared,admission='reuse-prefetch',n=1000000,warm=4,prefix=pref,split=('canonical_full_eval' if task=='subj' else 'full_eval'),adapter=adapter)
  run(name,cmd,a.gpu)
  assert hashlib.sha256((adapter/f'{task}.pt').read_bytes()).hexdigest()==sha
with (ROOT/f'audit/v26_quality_gpu{a.gpu}_done.json').open('x') as f:json.dump(dict(cells=cells,status='completed'),f)
