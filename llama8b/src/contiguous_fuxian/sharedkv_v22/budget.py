"""Physical-budget arithmetic, NOT a latency predictor or a runtime budget override."""
from __future__ import annotations
import argparse,json,math


def budget_frontier(*,layers=28,kv_heads=4,head_dim=128,block_tokens=16,base_blocks=77,
                    consumers=1,parameter_copies=True):
    if min(layers,kv_heads,head_dim,block_tokens,base_blocks)<=0 or not 0<=consumers<layers:
        raise ValueError('Invalid geometry/consumer count')
    block_bytes=2*block_tokens*kv_heads*head_dim*2
    original=layers*base_blocks*block_bytes
    # BF16 A/B/bias plus v21 transpose/bk-float packing. Head indices: 2*Hq*int64.
    base_reader=(2*kv_heads*head_dim*head_dim+2*kv_heads*head_dim)*2
    packed_extra=kv_heads*head_dim*head_dim*2+kv_heads*head_dim*4
    route=2*(kv_heads*7)*8   # Qwen2.5-7B only; exposed in returned assumptions.
    reader=base_reader+(packed_extra if parameter_copies else 0)+route
    source_count=layers-consumers
    extra=consumers*reader
    shared=source_count*base_blocks*block_bytes
    uniform=(original-extra)//(source_count*block_bytes)
    return {'assumptions':'Qwen2.5-7B: 7 Q heads/KV group; full 16-token blocks; FP16/BF16 both 2 bytes. Not total model memory.',
            'physical_sources':source_count,'baseline_payload_bytes':original,
            'same_retention_shared_payload_bytes':shared,'reader_and_route_bytes':extra,
            'net_persistent_saving_bytes':original-shared-extra,
            'iso_payload_plus_readers_max_uniform_blocks':uniform,
            'iso_budget_remainder_bytes':original-extra-uniform*source_count*block_bytes,
            'automatic_runtime_retention_change':False,
            'warning':'Changing retained IDs requires recalibration/revalidation and actual source/union accounting. This calculator does not change v2 keep_ratio guards.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--consumers',type=int,nargs='+',default=[1,2,3,6]);p.add_argument('--output')
    a=p.parse_args(argv);r=[budget_frontier(consumers=s) for s in a.consumers]
    text=json.dumps(r,indent=2)
    if a.output:
        from pathlib import Path
        path=Path(a.output)
        if path.exists():raise FileExistsError(path)
        path.write_text(text)
    print(text)
if __name__=='__main__':main()
