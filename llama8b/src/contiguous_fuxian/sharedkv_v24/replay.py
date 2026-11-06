"""Actual student replay from embeddings through all decoder layers.

Training uses a fixed, identity-produced sparse selection plan (no top-k
backprop). Private query/candidate KV is recomputed, not teacher-forced from
saved hidden states. Online evaluation still uses the unmodified selector.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from contiguous_fuxian.sharedkv_v23.block import layer_forward
from .records import assert_plan


def selected_bank(raw_prefix,row,sources,device,dtype):
    assert_plan(row,sources)
    bank={}
    for source in sorted(set(sources)):
        idx=torch.tensor(row['selection_by_layer'][source],device=device,dtype=torch.long)
        bank[source]=tuple(x.to(device=device,dtype=dtype).index_select(0,idx).unsqueeze(0)
                           for x in raw_prefix[source])
    return bank


def continuation_tokens(model,row,label,sources,bank,readers,*,checkpoint_layers=True,max_sequence=256):
    """Differentiable log p(y_j | query,y_<j), including the final label token.

    All labels are teacher-forced candidates, never the dataset's gold label.
    No T-1 vs T off-by-one: positions T-1 ... T+|y|-2 predict y_0 ... y_|y|-1.
    """
    if next(model.parameters()).device.type=='cuda' and 'teacher_backend' in row:
        from .faithful import continuation
        return continuation(model,row,label,sources,bank,readers,checkpoint_layers=checkpoint_layers,max_sequence=max_sequence)
    query=row['query_token_ids'];candidate=row['label_ids'][label]
    inputs=query+candidate[:-1]
    if len(inputs)>max_sequence:raise ValueError('Sequence exceeds explicit training memory limit; no truncation')
    device=next(model.parameters()).device
    ids=torch.tensor([inputs],device=device,dtype=torch.long)
    hidden=model.model.embed_tokens(ids)
    positions=torch.arange(row['prefix_tokens'],row['prefix_tokens']+len(inputs),device=device)[None]
    cos,sin=model.model.rotary_emb(hidden,positions)
    trainable_ids=[int(i) for i in readers]
    first=min(trainable_ids) if trainable_ids else len(sources)
    for lid,layer in enumerate(model.model.layers):
        pk,pv=bank[sources[lid]]
        reader=readers[str(lid)] if str(lid) in readers else None
        # Defaults bind loop variables correctly during backward recomputation.
        def step(x,layer=layer,pk=pk,pv=pv,reader=reader):
            return layer_forward(layer,x,pk,pv,cos,sin,
                                 reader.effective() if reader is not None else None)
        if checkpoint_layers and torch.is_grad_enabled() and lid>=first:
            hidden=checkpoint(step,hidden,use_reentrant=False)
        else:
            hidden=step(hidden)
    pos=torch.arange(len(query)-1,len(query)+len(candidate)-1,device=device)
    chosen=hidden.index_select(1,pos)
    logits=model.lm_head(model.model.norm(chosen)).float()[0]
    targets=torch.tensor(candidate,device=device,dtype=torch.long)
    return F.log_softmax(logits,dim=-1).gather(1,targets[:,None]).squeeze(1)


def score_vectors(token_lps):
    return torch.stack([x.mean() for x in token_lps])


def objective(student,teacher,*,temperature=1.,score_weight=1.,token_weight=.1,rank_weight=.25,margin_cap=.5):
    """Candidate KL + centered mean-score MSE + capped teacher-margin loss.

    softmax(mean token logprob) is an auxiliary candidate distribution, NOT
    an assertion that the underlying LM defines those class probabilities.
    """
    if len(student)!=len(teacher) or len(student)<2:raise ValueError('Candidate sets differ')
    if temperature<=0 or margin_cap<0:raise ValueError('Invalid loss settings')
    ss,tt=score_vectors(student),score_vectors(teacher).detach()
    tp=F.log_softmax(tt/temperature,dim=0);sp=F.log_softmax(ss/temperature,dim=0)
    kl=(tp.exp()*(tp-sp)).sum()*temperature**2
    centered=F.mse_loss(ss-ss.mean(),tt-tt.mean())
    token=sum(F.mse_loss(a,b.detach()) for a,b in zip(student,teacher))/len(student)
    top=int(tt.argmax());mask=torch.arange(len(tt),device=tt.device)!=top
    teacher_margin=(tt[top]-tt[mask]).clamp(min=0,max=margin_cap)
    student_margin=ss[top]-ss[mask]
    rank=(teacher_margin-student_margin).clamp_min(0).square().mean()
    loss=kl+score_weight*centered+token_weight*token+rank_weight*rank
    return loss,{'candidate_kl':kl,'centered_score_mse':centered,'token_mse':token,'margin_loss':rank}


def streamed_backward(compute,labels,teacher,loss_kwargs):
    """Exact chain-rule VJP with one candidate's model graph alive at a time.

    First pass obtains scalar/token targets; a tiny leaf graph yields dL/dlogp.
    Each branch is recomputed and immediately backpropagated. There is no
    optimizer step between passes. Requires deterministic eval-mode forwards.
    This saves memory, but uses more training forwards (reported explicitly).
    """
    with torch.no_grad():cached=[compute(label).detach() for label in labels]
    leaves=[x.clone().requires_grad_(True) for x in cached]
    loss,parts=objective(leaves,teacher,**loss_kwargs)
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite distillation loss')
    deriv=torch.autograd.grad(loss,leaves)
    max_recompute_diff=0.
    for label,old,weight in zip(labels,cached,deriv):
        values=compute(label)
        diff=float((values.detach()-old).abs().max())
        max_recompute_diff=max(max_recompute_diff,diff)
        if diff>1e-4:raise RuntimeError('Non-deterministic two-pass replay; no optimizer step performed')
        (values*weight.detach()).sum().backward()
    return float(loss.detach()),{k:float(v.detach()) for k,v in parts.items()},max_recompute_diff
