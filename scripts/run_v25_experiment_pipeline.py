
import json,time,hashlib
from v25_common import *
while True:
 t=(ROOT/'logs/v25_grid_driver.log').read_text()
 if 'Traceback (most recent call last)' in t:raise RuntimeError('Grid driver failed')
 if 'ALL V25 GRID TRAINING COMPLETE' in t:break
 time.sleep(10)
children=[]
for R,gpu in [(0,2),(4,3)]:
 jobs=[]
 if R==0:jobs.append(args('v25_dev_identity',split='explore768',n=768,gpu=gpu))
 for G in [26,24,22,20,16,14]:jobs.append(args(f'v25_dev_g{G}_r{R}_s24',G=G,R=R,split='explore768',n=768,gpu=gpu))
 children.append(suite(f'v25_development_r{R}',jobs,gpu))
wait_all(children)
base=rows('v25_dev_identity');assert len(base)==768
bc=sum(x['correct'] for x in base);counts={}
for G in [26,24,22,20,16,14]:
 for R in [0,4]:
  z=rows(f'v25_dev_g{G}_r{R}_s24');assert [x['uid'] for x in z]==[x['uid'] for x in base]
  counts[f'g{G}_r{R}']=sum(x['correct'] for x in z)
eligible=[G for G in [24,22,20,16,14] if max(counts[f'g{G}_r{R}'] for R in [0,4])/768>=bc/768-.01]
G=min(eligible) if eligible else 24
selected={'chosen_G':G,'R':[0,4],'seeds':[24,25,26],'baseline_dev_correct':bc,'dev_counts':counts,'point_screen_eligible_G':eligible,'fallback_used':not eligible,'final1536_evaluated':False,'seed24_timing_fixed':True,'scope':'Research selection, not noninferiority certification; both R retained.'}
with (ROOT/'audit/v25_structural_selection.json').open('x') as f:json.dump(selected,f,indent=2)
print('STRUCTURE SELECTED',json.dumps(selected),flush=True)
children=[launch(f'v25_replicates_r{R}',[PY,'-u','scripts/train_v25_replicates.py','--G',str(G),'--R',str(R),'--gpu',str(gpu)],gpu) for R,gpu in [(0,2),(4,3)]]
wait_all(children)
selected['checkpoint_sha256']={f'g{G}_r{R}_s{s}':hashlib.sha256((ROOT/f'assets/v25_g{G}_r{R}_s{s}/trec.pt').read_bytes()).hexdigest() for R in [0,4] for s in [24,25,26]}
selected['frozen_before_final1536']=True
with (ROOT/'audit/v25_frozen_selection.json').open('x') as f:json.dump(selected,f,indent=2)
children=[]
for R,gpu in [(0,2),(4,3)]:
 jobs=[]
 if R==0:
  jobs.append(args('v25_final_identity',split='final1536',n=1536,gpu=gpu))
  jobs.append(args('v25_test500_identity',n=500,gpu=gpu))
 for s in [24,25,26]:
  jobs.append(args(f'v25_final_g{G}_r{R}_s{s}',G=G,R=R,seed=s,split='final1536',n=1536,gpu=gpu))
 for g in [26,24,22,20,16,14]:jobs.append(args(f'v25_test500_g{g}_r{R}_s24',G=g,R=R,n=500,gpu=gpu))
 for s in [25,26]:jobs.append(args(f'v25_test500_g{G}_r{R}_s{s}',G=G,R=R,seed=s,n=500,gpu=gpu))
 children.append(suite(f'v25_final_quality_r{R}',jobs,gpu))
wait_all(children)
print('ALL V25 QUALITY COMPLETE',flush=True)
run('v25_formal_timing_driver',[PY,'-u','scripts/run_v25_timing.py'],3)
print('ALL V25 QUALITY AND TIMING COMPLETE',flush=True)
