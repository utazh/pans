
from pathlib import Path
import os,json,subprocess,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
PY='/home/panzihang/venvs/vllm-stable/bin/python'
MODEL='/data1/llm/Qwen/Qwen2.5-7B-Instruct'
def env(gpu):
 return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',PYTHONPATH='src:runtime_deps_v24')
def launch(name,cmd,gpu):
 telemetry=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu','--format=csv'],capture_output=True,text=True)
 with (ROOT/f'audit/{name}_environment.json').open('x') as f:json.dump(dict(time_unix=time.time(),gpu=gpu,nvidia_smi=telemetry.stdout,loadavg=os.getloadavg()),f)
 with (ROOT/f'configs/{name}_command.json').open('x') as f:json.dump(cmd,f,indent=2)
 with (ROOT/f'logs/{name}.log').open('x') as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env(gpu),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print('START',name,p.pid,flush=True);return p
def run(name,cmd,gpu):
 p=launch(name,cmd,gpu);start=time.time();code=p.wait()
 with (ROOT/f'audit/{name}_process.json').open('x') as f:json.dump(dict(returncode=code,wall_seconds=time.time()-start),f)
 if code:raise RuntimeError(f'{name} failed with {code}')
 print('DONE',name,flush=True)
def argv(name,task='trec',budget=25,gpu=3,shared=False,admission='original',n=8,warm=0,cold=None,diagnostic=False,proxy=False,prefix='p0',split='eval',adapter=None):
 store=ROOT/'runtime_data/prefix_store' if prefix=='p0' else ROOT/'runtime_data/v26_prefix_p1'
 index=ROOT/'assets/selector_index_k4_g32' if prefix=='p0' else ROOT/'assets/v26_selector_p1'
 cache=ROOT/f'runtime_data/v26_pcache_gpu{gpu}' if prefix=='p0' else ROOT/f'runtime_data/v26_p1_pcache_gpu{gpu}'
 bundle=ROOT/f'data/v26/{prefix}/{split}'
 if not bundle.exists() and task=='trec' and prefix=='p0':bundle=ROOT/'data/v25_splits/test500'
 a=[PY,'-u','scripts/run_evidence_v26.py','--admission',admission,'--audit-jsonl',str(ROOT/f'results/{name}.v26.jsonl')]
 if proxy:a+=['--repair-values','source-proxy']
 a+=['--','--model-path',MODEL,'--store-root',str(store),'--selector-index-dir',str(index),'--','--pcache-root',str(cache)]
 if cold:a+=['--cold','ssd-cold','--read-advice',cold,'--owned-cache-root',str(cache),'--cold-log',str(ROOT/f'results/{name}.cold.jsonl')]
 a+=['--','--mode','shared' if shared else 'identity','--backend','vllm2','--layout','hybrid','--metadata-cache','--stats',str(ROOT/f'results/{name}.extra.jsonl')]
 if shared:a+=['--packed-ops','--adapters',str(adapter or ROOT/'assets/v25_g16_r4_s24'),'--allow-candidate']
 if diagnostic:a+=['--diagnostic']
 a+=['--','--task',task,'--bundle-dir',str(bundle),'--samples',str(n),'--warmup',str(warm),'--max-tokens','1','--budget',str(budget),'--period','8','--profile','none','--output',str(ROOT/f'results/{name}')]
 return a
