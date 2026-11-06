#!/usr/bin/env python3
"""Complete P8 x four-budget x four-dataset evaluation, followed by separate diagnostic captures."""
import argparse,csv,hashlib,json,math,os,subprocess,time,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/panzihang/venvs/vllm-stable/bin/python'
NSYS=ROOT/'runtime_tools/nsight-2025.5.1/opt/nvidia/nsight-systems-cli/2025.5.1/target-linux-x64/nsys'
BUDGETS=[5,10,25,50]
TASKS=['sst2','subj','trec','rte']
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-root',type=Path,required=True)
p.add_argument('--profile-root',type=Path,required=True)
p.add_argument('--gpu',default='2')
p.add_argument('--resume',action='store_true')
a=p.parse_args();os.chdir(ROOT)
runroot=a.run_root.resolve();profroot=a.profile_root.resolve()
for path in [runroot,profroot]:path.relative_to(ROOT);path.mkdir(parents=True,exist_ok=a.resume)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
statepath=runroot/'status.json'
status=json.loads(statepath.read_text()) if a.resume and statepath.exists() else {'state':'starting','gpu':a.gpu,'period':8,'budgets':BUDGETS,'formal':{},'profiles':{},'started_at':time.time()}
status['orchestrator_pid']=os.getpid()
def save():
    status['updated_at']=time.time()
    tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2));tmp.replace(statepath)
def log(msg):print(time.strftime('%Y-%m-%d %H:%M:%S'),msg,flush=True)
expected={task:[json.loads(line)['uid'] for line in (ROOT/'data/paper_task_bundles_full_eval_strict'/f'{task}.jsonl').read_text().splitlines() if line] for task in TASKS}
(runroot/'task_uids.json').write_text(json.dumps(expected,indent=2))
sources=[x for d in ['src','scripts','tests','configs','vendor'] for x in (ROOT/d).rglob('*') if x.is_file() and '__pycache__' not in x.parts]
snapshot={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
snapfile=runroot/'source_sha256.json'
if a.resume and snapfile.exists():
    assert json.loads(snapfile.read_text())==snapshot,'Source changed since start'
else:snapfile.write_text(json.dumps(snapshot,indent=2))
def idle():
    while True:
        busy=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if not busy:return
        status['waiting_gpu_pids']=busy;save();time.sleep(30)
def command(task,budget,out,profile='none'):
    cmd=[PYTHON,str(ROOT/'scripts/run_round3.py'),'--task',task,'--budget',str(budget),'--period','8','--output',str(out)]
    if profile!='none':cmd+=['--profile',profile,'--samples','4','--warmup','4']
    return cmd
def validate(out,task,budget,profile=False):
    rows=[json.loads(line) for line in (out/'scored_records.jsonl').read_text().splitlines() if line]
    assert [r['uid'] for r in rows]==(expected[task][:4] if profile else expected[task]),'UID mismatch'
    assert len({r['uid'] for r in rows})==len(rows)
    for r in rows:
        assert r['exact_block_budget_consumed']==r['exact_block_budget_target'],'Budget mismatch'
        assert r['selector_calls']==4 and r['promixed_p8_decisions']==4 and r['promixed_mean_period']==8,'P8 mismatch'
        assert all(r[f'promixed_p{x}_decisions']==0 for x in [1,2,4]),'Unexpected P1/P2/P4'
        boundaries=[r[k] for k in ['logits_ready_ms','first_token_ready_ms','response_ready_ms','evaluation_ready_ms']]
        assert all(math.isfinite(v) for v in boundaries) and boundaries==sorted(boundaries)
    args=json.loads((out/'command.json').read_text())['args']
    assert args[args.index('--expected-keep-ratio')+1]==str(budget/100)
    assert args[args.index('--promixed-reuse-mode')+1]=='fixed'
    assert '--no-promixed-adaptive-coverage' in args and '--exact-layer-block-budget' in args
    s=json.loads((out/'summary.json').read_text())['tasks'][task]
    assert s['samples']==len(rows)
    return dict(samples=len(rows),accuracy=s['accuracy'],mean_ttft_ms=s['mean_ttft_ms'],p95_ttft_ms=s['p95_ttft_ms'],mean_response_ready_ms=s['mean_response_ready_ms'],mean_evaluation_ready_ms=s['mean_evaluation_ready_ms'],uid_order_exact=True,exact_budget_all=True,p8_all=True,timing_order_all=True)
def refresh_metrics():
    rows=[]
    for key,v in status['formal'].items():
        if v.get('state')=='complete':rows.append(dict(budget_pct=v['budget'],task=v['task'],**v['validation']))
    if rows:
        with (runroot/'metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (runroot/'validation.json').write_text(json.dumps({f"k{x['budget_pct']:03d}/{x['task']}":x for x in rows},indent=2))
save()
try:
    for budget in BUDGETS:
        for task in TASKS:
            key=f'k{budget:03d}/{task}';out=runroot/key
            if status['formal'].get(key,{}).get('state')=='complete':
                validate(out,task,budget);continue
            idle();status.pop('waiting_gpu_pids',None)
            status.update(state='formal',active=key)
            status['formal'][key]=dict(state='running',budget=budget,task=task,started_at=time.time());save()
            out.parent.mkdir(exist_ok=True);cmd=command(task,budget,out)
            logfile=runroot/f'k{budget:03d}_{task}.log'
            log('START formal '+key)
            with logfile.open('w') as f:
                child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
                status['child_pid']=child.pid;save();code=child.wait()
            assert code==0,f'Formal {key} failed: {code}'
            metrics=validate(out,task,budget)
            status['formal'][key].update(state='complete',finished_at=time.time(),validation=metrics);save();refresh_metrics()
            log('DONE formal '+key+' '+json.dumps(metrics))
    for budget in BUDGETS:
        for task in TASKS:
            for kind in ['nsys','viztracer']:
                key=f'k{budget:03d}/{task}/{kind}';base=profroot/f'k{budget:03d}'/task
                out=base/(kind+'_run');base.mkdir(parents=True,exist_ok=True)
                if status['profiles'].get(key,{}).get('state')=='complete':continue
                idle();status.pop('waiting_gpu_pids',None)
                assert os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>10*1024**3,'Less than 10 GiB free'
                status.update(state='profiling',active=key);status['profiles'][key]=dict(state='running',started_at=time.time());save()
                cmd=command(task,budget,out,kind)
                if kind=='nsys':
                    cmd=[str(NSYS),'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','-o',str(base/'nsys')]+cmd
                log('START profile '+key)
                with (base/(kind+'.log')).open('w') as f:
                    child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
                    status['child_pid']=child.pid;save();code=child.wait()
                (base/(kind+'_exitcode.txt')).write_text(str(code)+'\n')
                assert code==0,f'Profile {key} failed: {code}'
                v=validate(out,task,budget,True)
                requests=json.loads((out/'profile_requests.json').read_text())
                assert sum(bool(r['captured']) for r in requests)==4
                if kind=='nsys':
                    with (base/'nsys_export.log').open('w') as f:
                        subprocess.run([str(NSYS),'export','--type=sqlite','--output',str(base/'nsys.sqlite'),str(base/'nsys.nsys-rep')],check=True,stdout=f,stderr=subprocess.STDOUT)
                else:
                    meta=json.loads((out/'viztracer.json').read_text())
                    assert not meta.get('viztracer_metadata',{}).get('overflow',False),'VizTracer overflow'
                status['profiles'][key].update(state='complete',finished_at=time.time(),validation=v);save()
                log('DONE profile '+key)
    status.update(state='complete',finished_at=time.time());status.pop('active',None);status.pop('child_pid',None);save();log('ALL COMPLETE')
except BaseException as exc:
    status.update(state='failed',error=repr(exc),traceback=traceback.format_exc());save();log('FAILED '+repr(exc));raise
