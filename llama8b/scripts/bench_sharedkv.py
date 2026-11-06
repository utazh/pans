#!/usr/bin/env python3
"""Synthetic GPU transfer+attention microbenchmark, NOT model TTFT/quality.

Baseline and shared variants both use the same two-segment attention backend.
The source tensors are equal across the two synthetic layers, so the shared
variant can use identity readers. Include real learned-reader costs separately
in the model-level run_sharedkv.py experiment.
"""
from pathlib import Path
import argparse,json,sys,time,statistics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from contiguous_fuxian.sharedkv.core import Reader,segmented_attention


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix-tokens",type=int,default=8192)
    p.add_argument("--query-tokens",type=int,default=64)
    p.add_argument("--iterations",type=int,default=30)
    p.add_argument("--warmup",type=int,default=5)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA is required; no CPU timing fallback")
    if min(a.prefix_tokens,a.query_tokens,a.iterations)<=0:raise ValueError("Positive sizes required")
    if a.output.exists():raise FileExistsError(a.output)
    torch.manual_seed(0)
    hk,hq,d=4,28,128
    cpu=[torch.randn(1,a.prefix_tokens,hk,d,dtype=torch.float16).pin_memory() for _ in range(2)]
    q=[torch.randn(1,a.query_tokens,hq,d,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
    kt=torch.randn(1,a.query_tokens,hk,d,device="cuda",dtype=torch.bfloat16)
    vt=torch.randn_like(kt)
    eye=torch.eye(d,device="cuda",dtype=torch.bfloat16).repeat(hk,1,1)
    reader=Reader(eye,eye,torch.zeros(hk,d,device="cuda",dtype=torch.bfloat16),torch.zeros(hk,d,device="cuda",dtype=torch.bfloat16))
    def transfer():return tuple(x.to("cuda",non_blocking=True).to(torch.bfloat16) for x in cpu)
    def run(shared):
        bank=None
        for i in range(2):
            if bank is None or not shared:bank=transfer()
            out=segmented_attention(q[i],*bank,kt,vt,reader=reader if shared and i==1 else None,backend="flash2")
        return out
    measured={"independent":[],"shared":[]}
    with torch.inference_mode():
        for i in range(a.warmup):run(False);run(True)
        torch.cuda.synchronize()
        for i in range(a.iterations):
            for shared in ((False,True) if i%2==0 else (True,False)):
                torch.cuda.synchronize();start=time.perf_counter()
                run(shared);torch.cuda.synchronize()
                measured["shared" if shared else "independent"].append((time.perf_counter()-start)*1000)
    one=sum(x.numel()*x.element_size() for x in cpu)
    result={"type":"synthetic transfer+attention only; NOT model TTFT", "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "requested_cpu_to_gpu_payload_bytes":{"independent":2*one,"shared":one},
        "ms":{k:{"mean":statistics.mean(v),"median":statistics.median(v),"all":v} for k,v in measured.items()},
        "torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
        "warning":"No SSD/host gather/index/MLP/real Qwen accuracy here. Equal synthetic KV is not evidence of cross-layer similarity."}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
