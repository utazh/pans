"""Predeclared v24 ladder, frozen evaluation and serial paired timing."""
from pathlib import Path
import subprocess,os,json,time,concurrent.futures
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
def env(gpu):return dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH=str(r/'src')+':'+str(r/'runtime_deps_v24'))
def call(cmd,gpu,log):
 with (r/f'logs/{log}.log').open('x') as f:subprocess.run(cmd,cwd=r,env=env(gpu),stdout=f,stderr=subprocess.STDOUT,check=True)
def waitlog(name,token):
 while True:
  txt=(r/f'logs/{name}.log').read_text()
  if 'Traceback (most recent call last)' in txt:raise RuntimeError(name+' failed')
  if token in txt:return
  time.sleep(10)
def write(p,d):
 with (r/p).open('x') as f:json.dump(d,f,indent=2)
def quality_args(arm,name,split,n,gpu):
 a=json.loads((r/'configs/v24_group_dev_jobs.json').read_text())[0 if arm=='identity' else 1]
 for flag,value in [('--pcache-root',str(r/f'runtime_data/v24_pcache_gpu{gpu}')),('--stats',str(r/f'results/{name}.extra.jsonl')),('--bundle-dir',str(r/f'data/v24_splits/{split}')),('--samples',str(n)),('--output',str(r/f'results/{name}'))]:a[a.index(flag)+1]=value
 if arm!='identity':a[a.index('--adapters')+1]=str(r/f'assets/{arm}')
 return a
def runquality(arm,name,split,n,gpu):
 jobs=f'configs/{name}_job.json';write(jobs,[quality_args(arm,name,split,n,gpu)])
 call([py,'-u','scripts/run_layerkv_v24_suite.py',jobs],gpu,name)
def auditdev(arm,name):
 path=f'audit/{arm}_dev_quality.json'
 if not (r/path).exists():
  call([py,'scripts/audit_quality_v24.py','--baseline',str(r/'results/v24_dev_identity/scored_records.jsonl'),'--student',str(r/f'results/{name}/scored_records.jsonl'),'--output',str(r/path)],2,arm+'_dev_audit')
 return json.loads((r/path).read_text())
waitlog('v24_g26_dev','G26 DEVELOPMENT COMPLETE')
waitlog('v24_g26_training','G26 SCORE TRAINING COMPLETE')
for s in [24,25,26]:auditdev(f'v24_group_s{s}',f'v24_dev_group_s{s}')
stages=[];chosen=26;lastpassed=None
for G in [26,24,22,20,18,16]:
 if G!=26:
  def worker(seeds,gpu):
   for s in seeds:
    arm=f'v24_score_g{G}_s{s}';base=json.loads((r/'configs/v24_score_g26_s24_command.json').read_text())
    for flag,value in [('--worklog',str(r/f'audit/{arm}_compute.json')),('--seed',str(s)),('--plan',str(r/f'configs/v24_ladder/g{G}.json')),('--output',str(r/f'assets/{arm}/trec.pt'))]:base[base.index(flag)+1]=value
    base+=['--initial',str(r/f'assets/v24_score_g{G+2}_s{s}/trec.pt')]
    write(f'configs/{arm}_command.json',base)
    print('TRAIN',arm,'GPU',gpu,flush=True);call(base,gpu,arm)
    print('DEV',arm,flush=True);runquality(arm,f'v24_dev_score_g{G}_s{s}','dev768',768,gpu)
  with concurrent.futures.ThreadPoolExecutor(2) as ex:
   futures=[ex.submit(worker,[24,26],2),ex.submit(worker,[25],3)]
   for f in futures:f.result()
 reports={str(s):auditdev(f'v24_score_g{G}_s{s}',f'v24_dev_score_g{G}_s{s}') for s in [24,25,26]}
 passed=all(v['conservative_bound_passes'] for v in reports.values())
 stages.append({'G':G,'all_seed_dev_gate_passed':passed,'seeds':reports})
 write(f'audit/v24_stage_g{G}_decision.json',stages[-1])
 print('STAGE DECISION',G,passed,json.dumps(reports),flush=True)
 if not passed:break
 lastpassed=G;chosen=G
frozen={'chosen_G':chosen,'primary_seed':24,'last_all_seed_admitted_G':lastpassed,'stages':stages,'diagnostic_if_no_stage_passes':lastpassed is None,'final768_previously_unseen':True,'final_quality_variants':['identity']+[f'v24_score_g{chosen}_s{s}' for s in [24,25,26]],'reason':'Deepest stage passing all three predeclared development comparisons; G26 diagnostic retained if none passes','time':time.time()}
write('audit/v24_frozen_selection.json',frozen)
print('FROZEN',json.dumps(frozen),flush=True)
variants=frozen['final_quality_variants']
def finalworker(arms,gpu):
 for arm in arms:
  for split,n in [('final768',768),('test500',500)]:
   name=f'v24_{split}_{arm.removeprefix("v24_")}'
   print('FINAL QUALITY',name,flush=True);runquality(arm,name,split,n,gpu)
with concurrent.futures.ThreadPoolExecutor(2) as ex:
 futures=[ex.submit(finalworker,variants[::2],2),ex.submit(finalworker,variants[1::2],3)]
 for f in futures:f.result()
print('FROZEN QUALITY COMPLETE',flush=True)
primary=f'v24_score_g{chosen}_s24';timing=[]
for policy in ['warm','native','preserve']:
 for block in range(4):
  order=['identity','shared'] if block in (0,3) else ['shared','identity']
  for variant in order:
   name=f'v24_time_{policy}_b{block}_{variant}'
   a=quality_args('identity' if variant=='identity' else primary,name,'test500',32,3)
   if policy!='warm':
    pos=a.index('--');a[pos:pos]=['--cold','ssd-cold','--read-advice',policy,'--owned-cache-root',str(r/'runtime_data/v24_pcache_gpu3'),'--cold-log',str(r/f'results/{name}.cold.jsonl')]
   timing.append(a)
write('configs/v24_formal_timing_jobs.json',timing)
print('FORMAL TIMING START: no other v24 GPU jobs',flush=True)
write('configs/v24_formal_warm_jobs.json',timing[:8])
write('configs/v24_formal_cold_jobs.json',timing[8:])
call([py,'-u','scripts/run_layerkv_v24_suite.py','configs/v24_formal_warm_jobs.json'],3,'v24_formal_warm')
call([py,'-u','scripts/run_v24_cold_jobs.py',str(r/'configs/v24_formal_cold_jobs.json')],3,'v24_cold_resume')
print('LADDER QUALITY TIMING COMPLETE',flush=True)
