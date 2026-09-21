from pathlib import Path
import json,subprocess,os
r=Path('/home/panzihang/src/prism_layerkv');py='/home/panzihang/venvs/vllm-stable/bin/python'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='4')
for split in ('fit','val'):
 subprocess.run([py,'scripts/audit_group_ids_v23.py','--plans',str(r/f'runtime_data/v23_teacher_{split}/selection'),'--start','24','--end','28','--source','26','--native-layers','24','--output',str(r/f'audit/v23_group_ids_{split}.json')],cwd=r,env=env,check=True)
for name,extra in [('joint',[]),('independent',['--independent-layers']),('query',['--query-only'])]:
 cmd=[py,'-u','scripts/train_group_sharedkv_v23.py','--model-path','/data1/llm/Qwen/Qwen2.5-7B-Instruct','--store-root',str(r/'runtime_data/prefix_store'),'--task','trec','--teacher-fit',str(r/'runtime_data/v23_teacher_fit'),'--teacher-val',str(r/'runtime_data/v23_teacher_val'),'--student-fit',str(r/'runtime_data/v23_teacher_fit'),'--student-val',str(r/'runtime_data/v23_teacher_val'),'--start','24','--end','28','--source','26','--native-layers','24','--steps','200','--eval-every','50','--output',str(r/f'assets/v23_{name}/trec.pt'),*extra]
 print('START',name,flush=True)
 with (r/f'logs/v23_train_{name}.log').open('w') as f:subprocess.run(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,flush=True)
print('TRAINING COMPLETE',flush=True)
