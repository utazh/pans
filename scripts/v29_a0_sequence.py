"""v29 A0: Full-KV test references, SUBJ period diagnostic and identity dev baselines.
Quality only (no timing claims). GPU0/GPU1 each run one serial queue pinned to NUMA0 cores."""
from pathlib import Path
import json,os,subprocess,threading,time
R=Path("/home/panzihang/src/prism_layerkv");PY="/home/panzihang/venvs/vllm-stable/bin/python"
OUT=R/"studies/v29/a0";OUT.mkdir(parents=True,exist_ok=False)
N={"sst2":872,"subj":1003,"trec":500,"rte":277}
def job(name,task,budget,period,bundle,samples,pcache):
    o=OUT/"runs"/name
    return dict(name=name,task=task,budget=budget,period=period,bundle=str(bundle),samples=samples,argv=[PY,str(R/"scripts/run_evidence_v26.py"),"--admission","reuse-prefetch","--audit-jsonl",f"{o}.v26.jsonl","--",
      "--model-path","/data1/llm/Qwen/Qwen2.5-7B-Instruct","--store-root",str(R/"runtime_data/prefix_store"),"--selector-index-dir",str(R/"assets/selector_index_k4_g32"),"--",
      "--pcache-root",str(pcache),"--","--mode","identity","--backend","vllm2","--metadata-cache","--layout","hybrid","--stats",f"{o}.extra.jsonl","--",
      "--task",task,"--budget",str(budget),"--period",str(period),"--max-tokens","1","--warmup","0","--samples",str(samples),"--profile","none",
      "--bundle-dir",str(bundle),"--output",str(o)])
dev=R/"data/v29/p0/dev";full=lambda t:R/f"studies/v28_frozen/bundles/{t}_b50/eval"
queues={
 "0":dict(cores="0-7,32-39",pcache=R/"runtime_data/v27_pcache_gpu3",tasks=["sst2","subj"],diag=True),
 "1":dict(cores="8-15,40-47",pcache=R/"runtime_data/v27_pcache_gpu2",tasks=["trec","rte"],diag=False)}
plan={}
for g,q in queues.items():
    js=[job(f"fullkv_{t}",t,100,8,full(t),N[t],q["pcache"]) for t in q["tasks"]]
    if q["diag"]:js+=[job(f"subjdiag_b25_p{p}",   "subj",25,p,dev,500,q["pcache"]) for p in (4,1)]
    js+=[job(f"devI_{t}_b{b}",t,b,8,dev,500,q["pcache"]) for t in q["tasks"] for b in (5,10,25,50)]
    plan[g]=js
(OUT/"plan.json").write_text(json.dumps(dict(queues={g:dict(cores=q["cores"],pcache=str(q["pcache"])) for g,q in queues.items()},jobs=plan),indent=1))
(OUT/"runs").mkdir();(OUT/"logs").mkdir()
def run(g):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=g,PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",PYTHONPATH="src:runtime_deps_v24")
    for j in plan[g]:
        t0=time.time()
        with (OUT/"logs"/(j["name"]+".log")).open("x") as f:
            rc=subprocess.run(["taskset","-c",queues[g]["cores"]]+j["argv"],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        s=dict(name=j["name"],gpu=g,returncode=rc,seconds=round(time.time()-t0,1))
        with (OUT/"status.jsonl").open("a") as f:f.write(json.dumps(s)+"\n")
        print(json.dumps(s),flush=True)
ts=[threading.Thread(target=run,args=(g,)) for g in plan];[t.start() for t in ts];[t.join() for t in ts]
(OUT/"done.json").write_text(json.dumps(dict(finished=time.time())));print("A0_DONE",flush=True)
