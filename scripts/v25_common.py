
from pathlib import Path
import os,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
PY='/home/panzihang/venvs/vllm-stable/bin/python'
def env(gpu=None):
 e=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',PYTHONPATH='src:runtime_deps_v24')
 if gpu is not None:e['CUDA_VISIBLE_DEVICES']=str(gpu)
 return e
def launch(name,cmd,gpu):
 with (ROOT/f'configs/{name}_command.json').open('x') as f:json.dump(cmd,f,indent=2)
 with (ROOT/f'logs/{name}.log').open('x') as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env(gpu),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print('START',name,p.pid,flush=True);return p
def run(name,cmd,gpu):
 p=launch(name,cmd,gpu);code=p.wait()
 if code:raise RuntimeError(f'{name} failed with {code}; see its log')
 print('DONE',name,flush=True)
def train_cmd(G,R,seed):
 name=f'v25_g{G}_r{R}_s{seed}'
 return [PY,'-u','scripts/train_instrumented_v25.py','--native-blocks',str(R),'--worklog',str(ROOT/f'audit/{name}_compute.json'),'--','--model-path','/data1/llm/Qwen/Qwen2.5-7B-Instruct','--store-root',str(ROOT/'runtime_data/prefix_store'),'--task','trec','--teacher-fit',str(ROOT/'runtime_data/v24_teacher_fit'),'--teacher-val',str(ROOT/'runtime_data/v24_teacher_val'),'--plan',str(ROOT/f'configs/v25_explore/g{G}.json'),'--initial',str(ROOT/'assets/v24_score_g26_s24/trec.pt'),'--steps','200','--eval-every','25','--seed',str(seed),'--device','cuda','--output',str(ROOT/f'assets/{name}/trec.pt')]
def args(name,*,G=None,R=None,seed=24,split='test500',n=32,warm=0,gpu=3,cold=None,diagnostic=False,max_tokens=1,logits=False):
 a=['--','--pcache-root',str(ROOT/f'runtime_data/v25_pcache_gpu{gpu}')]
 if cold:
  a+=['--cold','ssd-cold','--read-advice',cold,'--owned-cache-root',str(ROOT/f'runtime_data/v25_pcache_gpu{gpu}'),'--cold-log',str(ROOT/f'results/{name}.cold.jsonl')]
 a+=['--','--mode','identity' if G is None else 'shared','--backend','vllm2','--layout','hybrid','--metadata-cache','--stats',str(ROOT/f'results/{name}.extra.jsonl')]
 if G is not None:a+=['--packed-ops','--adapters',str(ROOT/f'assets/v25_g{G}_r{R}_s{seed}'),'--allow-candidate']
 if diagnostic:a+=['--diagnostic']
 if logits:a+=['--logits',str(ROOT/f'runtime_data/{name}_logits')]
 a+=['--','--task','trec','--bundle-dir',str(ROOT/f'data/v25_splits/{split}'),'--samples',str(n),'--warmup',str(warm),'--max-tokens',str(max_tokens),'--budget','25','--period','8','--profile','none','--output',str(ROOT/f'results/{name}')]
 return a
def suite(name,jobs,gpu):
 p=ROOT/f'configs/{name}_jobs.json'
 with p.open('x') as f:json.dump([{'launcher':'run_native_prefix_v25.py','args':x} for x in jobs],f,indent=2)
 return launch(name,[PY,'-u','scripts/run_layerkv_v25_suite.py',str(p)],gpu)
def wait_all(children):
 codes=[p.wait() for p in children]
 if any(codes):raise RuntimeError(f'Child failures {codes}')
def rows(name):
 p=ROOT/f'results/{name}/scored_records.jsonl'
 z=[json.loads(x) for x in p.read_text().splitlines()]
 assert len({x['uid'] for x in z})==len(z)
 return z
