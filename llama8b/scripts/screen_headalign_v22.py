import subprocess,time,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for target,source in ((19,18),(27,26)):
 for mapping,name in (('search','map'),('identity','diag')):
  cmd=[sys.executable,'-u','scripts/compile_headalign_v22.py','--store-root','runtime_data/prefix_store','--task','trec','--teacher-fit','runtime_data/v21_teacher_fit','--teacher-val','runtime_data/v21_teacher_val','--student-fit','runtime_data/v21_teacher_fit','--student-val','runtime_data/v21_teacher_val','--source',str(source),'--target',str(target),'--mapping',mapping,'--steps','0','--device','cuda','--output',f'assets/v22_{name}{target}_s0/trec.pt']
  start=time.time(); print('START',cmd,flush=True);subprocess.run(cmd,cwd=root,check=True)
  print('DONE',target,mapping,time.time()-start,flush=True)
print('SCREENING COMPLETE',flush=True)
