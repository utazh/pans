"""Config-derived Qwen2 geometry. This is not a GPU/TP compatibility test."""
from __future__ import annotations

def inspect_config(c):
    if c.get('model_type')!='qwen2':raise ValueError('Only Qwen2-family is currently integrated')
    if c.get('use_sliding_window',False) or c.get('rope_scaling') is not None:
        raise ValueError('SWA/scaled RoPE requires a separate numerical audit')
    hq,hk,L,hidden=(int(c[x]) for x in ('num_attention_heads','num_key_value_heads','num_hidden_layers','hidden_size'))
    d=int(c.get('head_dim',hidden//hq))
    if min(hq,hk,L,hidden,d)<=0 or hq%hk or hidden!=hq*d:
        raise ValueError('Unsupported attention geometry')
    groups=hq//hk
    return dict(layers=L,q_heads=hq,kv_heads=hk,head_dim=d,q_per_kv=groups,
        probe_query_heads=[h*groups for h in range(hk)],selector_kv_head_ids=list(range(hk)),
        fp16_prefix_bytes_per_token=2*L*hk*d*2,
        reader_bf16_bytes_per_consumer=(2*hk*d*d+2*hk*d)*2,
        scope='Config check only. Rebuild KV/index, recollect calibration, refit readers, run CUDA parity. Single GPU only.')
