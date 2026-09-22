"""Reuse server-repaired v24 FA2 forward + mathematical LSE-aware backward.

Only prefix memory is changed. Private KV is recomputed per layer and kept
separate during every incremental candidate token. Source tensors stay native.
"""
from __future__ import annotations
import torch
from torch.utils.checkpoint import checkpoint
from contiguous_fuxian.sharedkv.core import apply_rope
from contiguous_fuxian.sharedkv_v21.ops import prepare_query,finish
from contiguous_fuxian.sharedkv_v24.faithful import attention
from contiguous_fuxian.sharedkv_v24.records import assert_plan
from .policy import partition

class Bank(dict):
    def __init__(self):
        super().__init__(); self.repairs={}; self.cuts={}


def selected_bank(raw,row,sources,device,dtype,policy):
    assert_plan(row,sources)
    bank=Bank()
    for source in sorted(set(sources)):
        ids=torch.tensor(row['selection_by_layer'][source],device=device,dtype=torch.long)
        bank[source]=tuple(x.to(device=device,dtype=dtype).index_select(0,ids).unsqueeze(0)
                           for x in raw[source])
    for lid,source in enumerate(sources):
        if lid==source:continue
        p=partition(row['selection_by_layer'][lid],row['prefix_tokens'],policy)
        bank.cuts[lid]=p.cut
        if p.native_ids:
            ids=torch.tensor(p.native_ids,device=device,dtype=torch.long)
            bank.repairs[lid]=tuple(x.to(device=device,dtype=dtype).index_select(0,ids).unsqueeze(0)
                                    for x in raw[lid])
    return bank


def decoder(model,ids,position,cache,sources,bank,readers,checkpoint_layers=True):
    hidden=model.model.embed_tokens(ids);b,t,_=hidden.shape
    pos=torch.arange(position,position+t,device=hidden.device)[None]
    cos,sin=model.model.rotary_emb(hidden,pos)
    first=min((int(i) for i in readers),default=len(sources));next_cache=[]
    for lid,layer in enumerate(model.model.layers):
        pk,pv=bank[sources[lid]]; prior=cache[lid] if cache else None
        reader=readers[str(lid)] if str(lid) in readers else None
        repair=bank.repairs.get(lid);cut=bank.cuts.get(lid,pk.shape[1])
        def step(x,layer=layer,pk=pk,pv=pv,prior=prior,reader=reader,repair=repair,cut=cut):
            attn=layer.self_attn;cfg=attn.config
            hq,hk,d=cfg.num_attention_heads,cfg.num_key_value_heads,attn.head_dim
            z=layer.input_layernorm(x)
            q=apply_rope(attn.q_proj(z).view(b,t,hq,d),cos,sin)
            k=apply_rope(attn.k_proj(z).view(b,t,hk,d),cos,sin)
            v=attn.v_proj(z).view(b,t,hk,d)
            tk,tv=(k,v) if prior is None else (torch.cat((prior[0],k),1),torch.cat((prior[1],v),1))
            if reader is None:
                out,_=attention(q,torch.cat((pk,tk),1),torch.cat((pv,tv),1),True)
            else:
                ek,ev=(tk,tv) if repair is None else (torch.cat((repair[0],tk),1),torch.cat((repair[1],tv),1))
                p=reader.effective()
                qp,offset=prepare_query(q,p.A.to(q.dtype).transpose(-1,-2).contiguous(),p.bk.float())
                up,lp=attention(qp,pk[:,:cut],pv[:,:cut],False)
                ut,lt=attention(q,ek,ev,True)
                out=finish(up,lp,ut,lt,offset,p.B.to(q.dtype).contiguous(),p.bv.to(q.dtype))
            mid=x+attn.o_proj(out.reshape(b,t,-1).contiguous())
            return mid+layer.mlp(layer.post_attention_layernorm(mid)),tk,tv
        if checkpoint_layers and torch.is_grad_enabled() and lid>=first:
            hidden,tk,tv=checkpoint(step,hidden,use_reentrant=False)
        else:hidden,tk,tv=step(hidden)
        next_cache.append((tk,tv))
    return model.lm_head(model.model.norm(hidden)),next_cache


def continuation(model,row,label,sources,bank,readers,*,checkpoint_layers=True,max_sequence=256):
    query=row['query_token_ids'];candidate=row['label_ids'][label]
    if len(query)+len(candidate)-1>max_sequence:
        raise ValueError('Sequence exceeds explicit memory limit; no truncation')
    if row.get('teacher_backend')!='vllm2' or row.get('teacher_layout')!='hybrid':
        raise ValueError('Reuse v24 teacher records captured on audited vllm2 hybrid path')
    device=next(model.parameters()).device
    logits,cache=decoder(model,torch.tensor([query],device=device),row['prefix_tokens'],None,sources,bank,readers,checkpoint_layers)
    scores=[logits[0,-1].float().log_softmax(-1)[candidate[0]]]
    for i,(token,target) in enumerate(zip(candidate,candidate[1:])):
        logits,cache=decoder(model,torch.tensor([[token]],device=device),row['prefix_tokens']+len(query)+i,
                             cache,sources,bank,readers,checkpoint_layers)
        scores.append(logits[0,-1].float().log_softmax(-1)[target])
    return torch.stack(scores)
