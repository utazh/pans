#!/usr/bin/env python3
"""Run a JSON list of launcher argument arrays; fresh Pcache store per job.
Reuse frozen model weights and content fingerprints only while all underlying
weight/config file stat signatures stay unchanged. Setup remains outside TTFT.
"""
import argparse,sys,runpy,importlib,gc,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
p=argparse.ArgumentParser();p.add_argument('jobs',type=Path);a=p.parse_args()
import torch
import contiguous_fuxian.flexgen_qwen_reprefill as runner
import contiguous_fuxian.quantized_key_index as index
load_model=runner._load_model_and_tokenizer
identity=index.model_identity
models={};identities={}
def signature(path):
    r=Path(path)
    paths=sorted(set(r.glob('*.safetensors'))|set(r.glob('pytorch_model*.bin'))|
                 set(r.glob('adapter_model*.bin'))|set(r.glob('*config.json')))
    return tuple((str(x),x.stat().st_size,x.stat().st_mtime_ns,x.stat().st_ino) for x in paths)
def cached_identity(path):
    key=(str(Path(path).resolve()),signature(path))
    if key not in identities: identities[key]=identity(path)
    return identities[key]
def cached_model(path,*,dtype,device):
    key=(str(Path(path).resolve()),dtype,device,signature(path))
    if key not in models: models[key]=load_model(path,dtype=dtype,device=device)
    return models[key]
index.model_identity=cached_identity
log=ROOT/'logs'/f'{a.jobs.stem}.progress.jsonl'
for job in json.loads(a.jobs.read_text()):
    importlib.reload(runner)
    runner._load_model_and_tokenizer=cached_model
    torch.cuda.reset_peak_memory_stats()
    started=time.time()
    print('START',job,flush=True)
    sys.argv=[str(ROOT/'scripts/run_sharedkv.py'),*job]
    runpy.run_path(str(ROOT/'scripts/run_sharedkv.py'),run_name='__main__')
    gc.collect();torch.cuda.empty_cache()
    with log.open('a') as f:f.write(json.dumps({'args':job,'wall_s':time.time()-started,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved()})+'\n')
    print('FINISHED',round(time.time()-started,2),flush=True)
print('SUITE COMPLETE',flush=True)
