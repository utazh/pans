#!/usr/bin/env python3
"""Run the four complete fixed-P1, uniform-25% workloads without profilers."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-root',type=Path,default=ROOT/'results'/('round1_k025_p1_'+time.strftime('%Y%m%d_%H%M%S')))
p.add_argument('--gpu',default='2')
a=p.parse_args()
os.chdir(ROOT)
runroot=a.run_root.resolve()
runroot.relative_to(ROOT)
runroot.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
python='/home/panzihang/venvs/vllm-stable/bin/python'
status={'state':'starting','gpu':a.gpu,'tasks':{}}
def save():
    path=runroot/'status.tmp';path.write_text(json.dumps(status,indent=2));path.replace(runroot/'status.json')
source_files=[x for folder in ['src','scripts','tests','configs','vendor'] for x in (ROOT/folder).rglob('*') if x.is_file() and '__pycache__' not in x.parts]
(runroot/'source_sha256.json').write_text(json.dumps({str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in source_files},indent=2))
save()
try:
    for task in ['rte','trec','sst2','subj']:
        busy=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if busy:raise RuntimeError(f'GPU {a.gpu} is occupied: {busy}')
        expected=[json.loads(x)['uid'] for x in (ROOT/'data/paper_task_bundles_full_eval_strict'/f'{task}.jsonl').read_text().splitlines() if x]
        (runroot/f'{task}_uids.json').write_text(json.dumps(expected,indent=2))
        status.update(state='running',active_task=task)
        status['tasks'][task]={'state':'running','expected_samples':len(expected)};save()
        with (runroot/f'{task}.log').open('w') as log:
            result=subprocess.run([python,str(ROOT/'scripts/run_round1.py'),'--task',task,'--output',str(runroot/task)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{task} exited with code {result.returncode}')
        rows=[json.loads(x) for x in (runroot/task/'scored_records.jsonl').read_text().splitlines()]
        if [x['uid'] for x in rows]!=expected:raise RuntimeError(f'{task} UID mismatch')
        for row in rows:
            if row['exact_block_budget_consumed']!=row['exact_block_budget_target']:raise RuntimeError('block budget mismatch')
            if row['selector_calls']!=28 or row['promixed_p1_decisions']!=28:raise RuntimeError('P1 not applied to all layers')
            if not row['logits_ready_ms']<=row['first_token_ready_ms']<=row['response_ready_ms']<=row['evaluation_ready_ms']:raise RuntimeError('timing boundary mismatch')
        status['tasks'][task].update(state='complete',samples=len(rows));save()
        print(task,'complete',len(rows),flush=True)
    status.update(state='complete');status.pop('active_task',None);save()
except BaseException as exc:
    status.update(state='failed',error=repr(exc));save();raise
