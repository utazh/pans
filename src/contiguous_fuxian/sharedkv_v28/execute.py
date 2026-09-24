"""Foreground, serial, immutable jobs. No polling, launch daemons or GPU killing."""
from __future__ import annotations
import json,subprocess,time,os,fcntl
from pathlib import Path
from .common import *
from .protocol import command,orders

def gpu_snapshot():
 try:
  with subprocess.Popen(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as process:
   try:return process.communicate(timeout=10)[0]
   except subprocess.TimeoutExpired:
    process.kill();return 'telemetry timeout: '+process.communicate()[0]
 except OSError as exc:return 'telemetry unavailable: '+str(exc)

def run_job(m,job,execute=False):
 root=Path(m['root']);spec=m['spec'];state=root/'jobs'/(job['id']+'.json')
 job_hash=digest(job);output=Path(job['output'])
 if state.exists():
  prior=json.loads(state.read_text())
  if prior['job_sha256']!=job_hash or prior['status']!='complete':
   raise RuntimeError('Job exists but differs/failed; preserve it and create a NEW study or attempt: '+str(state))
  assert_hashes(prior['result_sha256']);return prior
 print('JOB',job['id'],json.dumps(job['argv'],ensure_ascii=False),flush=True)
 if not execute:return None
 if output.exists():raise FileExistsError('Existing result without complete job record: '+str(output))
 assert_hashes(m['input_sha256'])
 log=root/'logs'/(job['id']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(spec['gpu']),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',
          PYTHONPATH='src:runtime_deps_v24')
 started=time.time();metadata=dict(job_sha256=job_hash,job=job,status='started',start_unix=started)
 if job['phase']=='timing':
  metadata['host_before']={'loadavg':list(os.getloadavg()),'gpu_snapshot':gpu_snapshot()}

 with log.open('x') as f:
  result=subprocess.run(job['argv'],cwd=spec['repo'],env=env,stdout=f,stderr=subprocess.STDOUT,check=False)
 metadata.update(returncode=result.returncode,process_seconds=time.time()-started)
 if job['phase']=='timing':
  metadata['host_after']={'loadavg':list(os.getloadavg()),'gpu_snapshot':gpu_snapshot()}

 if result.returncode:
  metadata['status']='failed';write_json(state,metadata);raise RuntimeError('Job failed; see '+str(log))
 records=output/('quality_records_v27.jsonl' if job['quality_only'] else 'scored_records.jsonl')
 rows=read_jsonl(records)
 expected=spec['block_size'] if job['block'] is not None else (16 if job['phase'].startswith('parity') else TASKS[job['task']])
 bundle_dir=Path(job['argv'][job['argv'].index('--bundle-dir')+1])
 requested=read_jsonl(bundle_dir/(job['task']+'.jsonl'))
 if [str(r['uid']) for r in rows]!=[str(r['uid']) for r in requested]:raise RuntimeError('Returned UID order differs from frozen input')
 if len(rows)!=expected or len({str(r['uid']) for r in rows})!=expected:
  metadata['status']='invalid_row_count';write_json(state,metadata);raise RuntimeError('Full result/UID count mismatch')
 paths=[records,log]
 raw=output/'scored_records.jsonl'
 if raw!=records:paths.append(raw)
 if job.get('logits_dir'):
  paths.extend(sorted(Path(job['logits_dir']).rglob('*.pt')))
 if job.get('representation'):paths.append(Path(job['representation']))
 for suffix in ('.v26.jsonl.effective_cli.json','.extra.jsonl','.extra.jsonl.v25.jsonl'):
  extra=Path(str(output)+suffix)
  if extra.is_file():paths.append(extra)
 if job['condition']!='warm':
  cold=Path(str(output)+'.cold.jsonl');cert=read_jsonl(cold)
  if len(cert)!=expected or any(x.get('status')!='completed' or x.get('page_after',{}).get('resident')!=0 for x in cert):
   metadata['status']='invalid_cold';write_json(state,metadata);raise RuntimeError('Cold certification missing/failed')
  paths.append(cold)
 metadata.update(status='complete',result_sha256={str(p):sha(p) for p in paths},n=len(rows))
 write_json(state,metadata);return metadata

def score_parity(a,b):
 A={str(x['uid']):x for x in read_jsonl(Path(a)/'scored_records.jsonl')}
 B={str(x['uid']):x for x in read_jsonl(Path(b)/'scored_records.jsonl')}
 if set(A)!=set(B) or not A:raise ValueError('Unpaired parity data')
 worst=0.
 for uid in A:
  x=A[uid]['label_candidate_mean_logprobs'];y=B[uid]['label_candidate_mean_logprobs']
  if set(x)!=set(y):raise ValueError('Candidate set differs')
  for k in x:
   import math
   if not math.isfinite(x[k]) or not math.isfinite(y[k]):raise ValueError('Nonfinite parity score')
   worst=max(worst,abs(x[k]-y[k]))
 return dict(max_label_score_difference=worst,threshold=.05,passed=worst<.05,n=len(A),
             scope='Full-rank functional parity on fixed identity selection; not approximate model quality')

def quality_jobs(m,c):
 return [command(m,c,a,'quality') for a in ('I','R4','R0')]

def run_external(m,c,execute):
 root=Path(m['root']);rep=root/'representations'/c['id'];gate=root/'gates'/(c['id']+'.json')
 needed=['clf_full64.pt','direct_identity.pt']
 if any(not (rep/x).exists() for x in needed):raise FileNotFoundError('Compile both baselines first: '+str(rep))
 identity_logits=root/'parity_logits'/c['id']/'I'
 ji=command(m,c,'I','parity',logits=identity_logits);run_job(m,ji,execute)
 parity={};artifacts={str(p):sha(p) for p in sorted(rep.glob('*.pt'))}
 for tag in ('direct_identity','clf_full64'):
  j=command(m,c,tag,'parity',representation=rep/(tag+'.pt'),selection=identity_logits)
  run_job(m,j,execute)
  if execute:parity[tag]=score_parity(ji['output'],j['output'])
 if execute:
  value=dict(parity=parity,artifact_sha256=artifacts,all_structural_gates_passed=all(p['passed'] for p in parity.values()))
  if gate.exists():
   old=json.loads(gate.read_text())
   if old!=value:raise ValueError('Gate or artifacts changed; use new study')
  else:write_json(gate,value)
  if not value['all_structural_gates_passed']:raise RuntimeError('Original .05 structural gate failed; do not continue/relax it')
 for p in sorted(rep.glob('*.pt')):
  if p.stem in ('direct_identity','clf_full64'):continue
  if not (p.stem.startswith('kvsharer_g') or p.stem.startswith('clf_')):continue
  if execute:assert_hashes(artifacts)
  j=command(m,c,p.stem,'external_quality',representation=p)
  run_job(m,j,execute)

def run_timing(m,c,execute):
 root=Path(m['root']);spec=m['spec']
 # For comparable inference paths, complete ALL three full-quality runs first.
 for a in ('I','R4','R0'):
  jid=command(m,c,a,'quality')['id'];state=root/'jobs'/(jid+'.json')
  if execute and (not state.exists() or json.loads(state.read_text())['status']!='complete'):
   raise ValueError('Complete all main quality arms before timing; no score-based selection implied')
 for condition in spec['conditions']:
  for b,order in enumerate(orders(spec['blocks'])):
   for pos,arm in enumerate(order):
    job=command(m,c,arm,'timing',block=b,condition=condition);job['position']=pos
    run_job(m,job,execute)

def execute_stage(manifest,stage,cells=None,execute=False):
 m=json.loads(Path(manifest).read_text());check(m['spec']['repo']);assert_hashes(m['input_sha256'])
 selected=[c for c in m['cells'] if not cells or c['id'] in cells]
 if cells and {c['id'] for c in selected}!=set(cells):raise ValueError('Unknown cell ID')
 lock=Path(m['root'])/'.foreground.lock'
 with lock.open('a') as stream:
  fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for c in selected:
   if stage=='quality':
    for job in quality_jobs(m,c):run_job(m,job,execute)
   elif stage=='baseline-quality':run_external(m,c,execute)
   elif stage=='timing':run_timing(m,c,execute)
   else:raise ValueError('Unknown stage')
