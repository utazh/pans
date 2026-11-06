#!/usr/bin/env python3
"""Foreground serial execution; outputs/failures preserved; no background scheduler."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contiguous_fuxian.sharedkv_v26.study import sha256
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--execute',action='store_true')
p.add_argument('--max-jobs',type=int,default=0);a=p.parse_args()
path=Path(a.manifest).resolve();s=json.loads(path.read_text())
if s.get('format')!='pans-frozen-evidence-study-v26':raise ValueError('Not a v26 study')
if not a.execute:
 print(json.dumps({'pending_jobs':len(s['jobs']),'note':'Planning only; supply --execute to run serially in the foreground.'}));sys.exit(0)
root=path.parent;logdir=root/'execution';logdir.mkdir(exist_ok=True)
count=0
for job in s['jobs']:
 status=logdir/(job['job_id']+'.json')
 if status.exists():
  old=json.loads(status.read_text())
  if old.get('returncode')==0:continue
  raise RuntimeError(f'Previous failed arm retained: {status}; prepare a new study instead of overwriting')
 adapter=Path(s['spec']['adapter_dir'])/(s['spec']['task']+'.pt')
 if sha256(adapter)!=s['checkpoint_sha256']:raise ValueError('Checkpoint changed')
 bundle=root/'bundles'/f"b{job['block']:02d}"/f"{s['spec']['task']}.jsonl"
 if sha256(bundle)!=job['bundle_sha256']:raise ValueError('Timing query block changed')
 if Path(job['output']).exists():raise FileExistsError(job['output'])
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(s['spec']['gpu']),OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
 print('START',job['job_id'],flush=True)
 start=time.time()
 telemetry=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu','--format=csv'],capture_output=True,text=True).stdout
 with (logdir/(job['job_id']+'.log')).open('x',encoding='utf8') as f:
  proc=subprocess.run(job['argv'],cwd=job['cwd'],env=env,stdout=f,stderr=subprocess.STDOUT,check=False)
 with status.open('x') as f:json.dump({'job':job['job_id'],'argv':job['argv'],'returncode':proc.returncode,
     'wall_seconds_including_loading':time.time()-start,'nvidia_smi_before':telemetry,'host_loadavg_after':os.getloadavg(),'manifest_sha256':sha256(path)},f,indent=2)
 if proc.returncode:raise RuntimeError(f'Arm failed; log: {logdir/job["job_id"]}')
 print('DONE',job['job_id'],flush=True)
 count+=1
 if a.max_jobs and count>=a.max_jobs:break
