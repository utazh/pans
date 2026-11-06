from pathlib import Path
import json
r=Path('/home/panzihang/src/prism_layerkv')
x=json.loads((r/'assets/v23_query/trec.pt.report.json').read_text())['args']
cmd=['/home/panzihang/venvs/vllm-stable/bin/python','-u','scripts/train_group_sharedkv_v23.py']
for k in ['model_path','store_root','task','teacher_fit','teacher_val','student_fit','student_val','start','end','source','native_layers']:
 cmd+=['--'+k.replace('_','-'),str(x[k])]
cmd+=['--initial-bundle',str(r/'assets/v23_query/trec.pt'),'--steps','0','--output',str(r/'assets/v23_query_full_validation/trec.pt')]
(r/'configs/v23_query_full_validation_command.json').write_text(json.dumps(cmd))
import subprocess,os
with (r/'logs/v23_query_full_validation.log').open('w') as f:
 subprocess.run(cmd,cwd=r,env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
print('COMMON FULL-SUFFIX VALIDATION COMPLETE')
