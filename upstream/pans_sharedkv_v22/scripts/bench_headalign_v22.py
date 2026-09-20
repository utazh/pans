#!/usr/bin/env python3
"""GPU-resident permutation overhead only. Not TTFT, H2D or task quality evidence."""
from pathlib import Path
import argparse,json,time,statistics,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',required=True);p.add_argument('--target',type=int,required=True)
    p.add_argument('--backend',choices=('flash2','vllm2'),default='vllm2')
    p.add_argument('--prefix-tokens',type=int,default=1228);p.add_argument('--query-tokens',type=int,default=41)
    p.add_argument('--iterations',type=int,default=100);p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--output',required=True);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; CPU timings are not substituted')
    if min(a.prefix_tokens,a.query_tokens,a.iterations,a.warmup)<=0:raise ValueError('Sizes/counts must be positive')
    if Path(a.output).exists():raise FileExistsError(a.output)
    from contiguous_fuxian.sharedkv.io import load_safe
    from contiguous_fuxian.sharedkv.core import Reader
    from contiguous_fuxian.sharedkv_v21.ops import pack,dispatch
    from contiguous_fuxian.sharedkv_v21.backend import LengthPool,_ACTIVE_POOL
    from contiguous_fuxian.sharedkv_v22.ops import prepare_route,mapped_attention
    bundle=load_safe(a.adapter);g=bundle['geometry'];pi=bundle.get('head_maps',{}).get(str(a.target),list(range(g['kv_heads'])))
    r=Reader.from_state_dict(bundle['readers'][str(a.target)]).to(device='cuda',dtype=torch.bfloat16)
    packed=pack(r);route=prepare_route(pi,q_heads=g['q_heads'],device='cuda')
    torch.manual_seed(22)
    q=torch.randn(1,a.query_tokens,g['q_heads'],g['head_dim'],device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,a.prefix_tokens,g['kv_heads'],g['head_dim'],device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    tk=torch.randn(1,a.query_tokens,g['kv_heads'],g['head_dim'],device='cuda',dtype=torch.bfloat16);tv=torch.randn_like(tk)
    functions={'same_index_operator':lambda:dispatch(q,k,v,tk,tv,reader=r,packed=packed,backend=a.backend),
               'mapped_operator':lambda:mapped_attention(q,k,v,tk,tv,reader=r,route=route,packed=packed,backend=a.backend)}
    rows={key:[] for key in functions};token=_ACTIVE_POOL.set(LengthPool())
    try:
        with torch.inference_mode():
            for _ in range(a.warmup):
                for f in functions.values():f()
            for it in range(a.iterations):
                order=list(functions) if it%2==0 else list(reversed(functions))
                for name in order:
                    torch.cuda.synchronize();start=time.perf_counter();functions[name]();torch.cuda.synchronize()
                    rows[name].append((time.perf_counter()-start)*1000)
    finally:_ACTIVE_POOL.reset(token)
    result={'args':vars(a),'head_map':pi,'median_host_wall_ms':{k:statistics.median(v) for k,v in rows.items()},
            'raw_host_wall_ms':rows,'warning':'Synthetic resident tensors; the two paths compute different functions for nonidentity maps. Timing only, not quality or TTFT.'}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result['median_host_wall_ms'],indent=2))
if __name__=='__main__':main()
