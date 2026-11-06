
from pathlib import Path
import os,subprocess,json,argparse
p=argparse.ArgumentParser();p.add_argument('--R',type=int,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',PYTHONPATH='src:runtime_deps_v24')
for G in [24,22,26,20,16,14]:
 name=f'v25_g{G}_r{a.R}_s24'
 cmd=[py,'-u','scripts/train_instrumented_v25.py','--native-blocks',str(a.R),'--worklog',str(r/f'audit/{name}_compute.json'),'--','--model-path','/data1/llm/Qwen/Qwen2.5-7B-Instruct','--store-root',str(r/'runtime_data/prefix_store'),'--task','trec','--teacher-fit',str(r/'runtime_data/v24_teacher_fit'),'--teacher-val',str(r/'runtime_data/v24_teacher_val'),'--plan',str(r/f'configs/v25_explore/g{G}.json'),'--initial',str(r/'assets/v24_score_g26_s24/trec.pt'),'--steps','200','--eval-every','25','--seed','24','--device','cuda','--output',str(r/f'assets/{name}/trec.pt')]
 with (r/f'configs/{name}_command.json').open('x') as f:json.dump(cmd,f,indent=2)
 print('START',name,flush=True)
 with (r/f'logs/{name}.log').open('x') as f:subprocess.run(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,flush=True)
print('GRID WORKER COMPLETE',a.R,flush=True)
