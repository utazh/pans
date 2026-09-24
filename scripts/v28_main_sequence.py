from pathlib import Path
import subprocess,os,json,time
r=Path('/home/panzihang/src/prism_layerkv');os.chdir(r)
py='/home/panzihang/venvs/vllm-stable/bin/python'
env=dict(os.environ,PYTHONPATH='src:runtime_deps_v24',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
manifest='studies/v28_frozen/manifest.json'
commands=[[py,'-u','scripts/matrix_v28.py','run','--manifest',manifest,'--stage','quality','--cells','trec_b25','--execute'],[py,'-u','scripts/matrix_v28.py','run','--manifest',manifest,'--stage','quality','--execute']]
for i,cmd in enumerate(commands):
 print('STAGE',i,cmd,flush=True);start=time.time()
 result=subprocess.run(cmd,env=env)
 (r/f'audit/v28_main_stage_{i}.json').write_text(json.dumps(dict(argv=cmd,returncode=result.returncode,seconds=time.time()-start)))
 if result.returncode:raise SystemExit(result.returncode)
(r/'audit/v28_main_done.json').write_text(json.dumps(dict(completed=time.time())))
