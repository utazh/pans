
from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='4',PYTHONPATH=str(r/'runtime_deps_v24')+':'+str(r/'src'))
for seed in [24,25,26]:
 name=f'v24_group_s{seed}'
 args=['--model-path','/data1/llm/Qwen/Qwen2.5-7B-Instruct','--store-root',str(r/'runtime_data/prefix_store'),'--task','trec','--teacher-fit',str(r/'runtime_data/v23_teacher_fit'),'--teacher-val',str(r/'runtime_data/v23_teacher_val'),'--student-fit',str(r/'runtime_data/v23_teacher_fit'),'--student-val',str(r/'runtime_data/v23_teacher_val'),'--start','24','--end','28','--source','26','--native-layers','24','--steps','200','--eval-every','25','--seed',str(seed),'--output',str(r/f'assets/{name}/trec.pt')]
 cmd=[py,'-u','scripts/train_instrumented_v24.py','--kind','group','--worklog',str(r/f'audit/{name}_compute.json'),'--',*args]
 (r/f'configs/{name}_command.json').write_text(json.dumps(cmd,indent=2))
 print('START',name,flush=True)
 with (r/f'logs/{name}.log').open('x') as f:subprocess.run(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,flush=True)
print('GROUP CONTROLS COMPLETE',flush=True)
