import json,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parents[1]
for target,source in ((19,18),(27,26)):
 a=json.loads((r/f'assets/v22_map{target}_s0/trec.report.json').read_text())
 b=json.loads((r/f'assets/v22_diag{target}_s0/trec.report.json').read_text())
 if a['head_map']==[0,1,2,3] or a['final']['output_relative_rmse']>=b['final']['output_relative_rmse']:
  print('STOP_NO_LOCAL_SIGNAL',target,flush=True);continue
 for mapping,name in (('search','map'),('identity','diag')):
  cmd=[sys.executable,'-u','scripts/compile_headalign_v22.py','--store-root','runtime_data/prefix_store','--task','trec','--teacher-fit','runtime_data/v21_teacher_fit','--teacher-val','runtime_data/v21_teacher_val','--student-fit','runtime_data/v21_teacher_fit','--student-val','runtime_data/v21_teacher_val','--source',str(source),'--target',str(target),'--mapping',mapping,'--steps','200','--eval-every','20','--lr','0.0001','--bf16-ste','--device','cuda','--output',f'assets/v22_{name}{target}_s200/trec.pt']
  start=time.time();print('START',cmd,flush=True);subprocess.run(cmd,cwd=r,check=True);print('DONE',target,mapping,time.time()-start,flush=True)
print('TRAINING COMPLETE',flush=True)
