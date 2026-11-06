#!/usr/bin/env python3
"""CUDA-resident attention microbenchmark. NOT model TTFT or H2D measurement.

Use the ACTUAL selected-prefix/query geometry and optionally a real reader.
CUDA-event and synchronized host wall times are reported independently.
"""
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from contiguous_fuxian.sharedkv.core import Reader,segmented_attention,attention_with_lse
from contiguous_fuxian.sharedkv.io import load_safe
from contiguous_fuxian.sharedkv_v21.ops import pack,prepare_query,finish,packed_attention,native_attention


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selected-tokens",type=int,default=2048);p.add_argument("--query-tokens",type=int,default=64)
    p.add_argument("--iterations",type=int,default=40);p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--backend",choices=["flash2","vllm2"],default="vllm2");p.add_argument("--adapter");p.add_argument("--consumer",type=int);p.add_argument("--compile",action="store_true")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required; no CPU latency fallback")
    if min(a.selected_tokens,a.query_tokens,a.iterations,a.warmup)<=0:raise ValueError("Positive sizes required")
    if Path(a.output).exists():raise FileExistsError(a.output)
    torch.manual_seed(13);h,hq,d=4,28,128
    if a.adapter:
        bundle=load_safe(a.adapter)
        if a.consumer is None:raise ValueError("--consumer required")
        reader=Reader.from_state_dict(bundle["readers"][str(a.consumer)]).to(device="cuda",dtype=torch.bfloat16)
        h=reader.A.shape[0];d=reader.A.shape[-1];hq=bundle["geometry"]["q_heads"]
    else:
        eye=torch.eye(d).repeat(h,1,1)
        reader=Reader(eye+.01*torch.randn_like(eye),eye+.01*torch.randn_like(eye),
                      .01*torch.randn(h,d),.01*torch.randn(h,d)).to(device="cuda",dtype=torch.bfloat16)
    packed=pack(reader)
    def rand(n,heads):return torch.randn(1,n,heads,d,device="cuda",dtype=torch.bfloat16)*.3
    q=rand(a.query_tokens,hq);pk,pv=rand(a.selected_tokens,h),rand(a.selected_tokens,h)
    tk,tv=rand(a.query_tokens,h),rand(a.query_tokens,h)
    with torch.inference_mode():
        qr,off=prepare_query(q,packed.A_transposed,packed.bk_float)
        up,lp=attention_with_lse(qr,pk,pv,backend=a.backend)
        ut,lt=attention_with_lse(q,tk,tv,causal=True,backend=a.backend)
        compiled_ms=None
        cases={
            "native_cat_one_attention":lambda:native_attention(q,pk,pv,tk,tv,backend=a.backend),
            "identity_two_attention":lambda:segmented_attention(q,pk,pv,tk,tv,backend=a.backend),
            "v1_reader_two_attention":lambda:segmented_attention(q,pk,pv,tk,tv,reader=reader,backend=a.backend),
            "v21_reader_two_attention":lambda:packed_attention(q,pk,pv,tk,tv,packed,backend=a.backend),
            "q_transform_offset_only":lambda:prepare_query(q,packed.A_transposed,packed.bk_float),
            "prefix_attention_only":lambda:attention_with_lse(qr,pk,pv,backend=a.backend),
            "private_attention_only":lambda:attention_with_lse(q,tk,tv,causal=True,backend=a.backend),
            "v_transform_merge_only":lambda:finish(up,lp,ut,lt,off,packed.B,packed.bv)}
        from contiguous_fuxian.sharedkv_v21.backend import LengthPool,_ACTIVE_POOL
        pool=LengthPool()
        def cached_reader():
            token=_ACTIVE_POOL.set(pool)
            try:return packed_attention(q,pk,pv,tk,tv,packed,backend=a.backend)
            finally:_ACTIVE_POOL.reset(token)
        cases["v21_reader_cached_metadata"]=cached_reader
        if a.compile:
            pq=torch.compile(prepare_query,fullgraph=True);fv=torch.compile(finish,fullgraph=True)
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(3):pq(q,packed.A_transposed,packed.bk_float);fv(up,lp,ut,lt,off,packed.B,packed.bv)
            torch.cuda.synchronize();compiled_ms=1000*(time.perf_counter()-start)
            cases["compiled_q_only"]=lambda:pq(q,packed.A_transposed,packed.bk_float)
            cases["compiled_v_merge_only"]=lambda:fv(up,lp,ut,lt,off,packed.B,packed.bv)
        baseline=cases["v1_reader_two_attention"]();new=cases["v21_reader_two_attention"]()
        parity=float((baseline.float()-new.float()).abs().max())
        if not torch.allclose(baseline,new,atol=.02,rtol=.02):raise ValueError("Reader backend parity failed")
        for fn in cases.values():
            for _ in range(a.warmup):fn()
        torch.cuda.synchronize()
        result={name:{"wall_ms":[],"cuda_ms":[]} for name in cases}
        for i in range(a.iterations):
            names=list(cases) if i%2==0 else list(cases)[::-1]
            for name in names:
                torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin=time.perf_counter();start.record();cases[name]();end.record();end.synchronize()
                result[name]["wall_ms"].append(1000*(time.perf_counter()-begin))
                result[name]["cuda_ms"].append(start.elapsed_time(end))
        for values in result.values():
            for kind in ("wall_ms","cuda_ms"):
                x=torch.tensor(values[kind]);values[kind+"_median"]=float(x.median());values[kind+"_p95"]=float(torch.quantile(x,.95))
    import importlib.metadata
    backend_version=importlib.metadata.version("vllm" if a.backend=="vllm2" else "flash-attn")
    report={"scope":"synthetic/resident microbench; not TTFT, not H2D, not real Qwen quality",
        "args":vars(a),"torch":torch.__version__,"backend_version":backend_version,
        "gpu":torch.cuda.get_device_name(),"reader_parity_max_abs":parity,
        "compile_and_probe_ms_excluded":compiled_ms,"results":result,
        "warning":"Component times cannot be summed to infer overlapped model critical path"}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2))

if __name__=="__main__":main()
