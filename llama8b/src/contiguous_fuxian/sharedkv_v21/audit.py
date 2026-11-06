"""Diagnostics, not invented causal attributions or guarantees."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import torch
from .traces import load_traces,paired
from contiguous_fuxian.sharedkv.io import read_prefix


def relative_rmse(x,y):
    x,y=x.double(),y.double()
    return float(((x-y).square().sum()/y.square().sum().clamp_min(1e-30)).sqrt())


def logit_span_bound(source_keys, teacher_logits):
    """Optimistic lower bound for UNWEIGHTED logit least-squares only.

    For each query independently optimize arbitrary read vector + intercept.
    This is MORE expressive than a common A,bk. NOT a bound on softmax KL,
    output error or task accuracy. If N <= d+1 it can be vacuous (zero).
    source_keys [N,D], teacher_logits [M,N].
    """
    x,y=source_keys.double(),teacher_logits.double()
    x=torch.cat((x,torch.ones(x.shape[0],1,dtype=x.dtype,device=x.device)),1)
    u,s,_=torch.linalg.svd(x,full_matrices=False)
    tol=max(x.shape)*torch.finfo(x.dtype).eps*s.max()
    rank=int((s>tol).sum());basis=u[:,:rank]
    residual=y-(y@basis)@basis.T
    centered=y-y.mean(-1,keepdim=True)
    return {"rank":rank,"tokens":x.shape[0],"relative_to_centered_logit_energy":
        float((residual.square().sum()/centered.square().sum().clamp_min(1e-30)).sqrt()),
        "rmse":float(residual.square().mean().sqrt()),"potentially_vacuous":x.shape[0]<=x.shape[1]}


def audit_traces(teacher,student):
    layers=sorted(set.intersection(*(set(r["layers"]) for r in list(teacher.values())+list(student.values()))))
    result={}
    for lid in layers:
        rows=paired(teacher,student,lid,require_same_ids=False)
        values={k:[] for k in ("q","hidden_in","pre_mlp","hidden_out","attention_output")};jaccards=[]
        for t,s in rows:
            a,b=t["layers"][lid],s["layers"][lid]
            for name in values:values[name].append(relative_rmse(b[name],a[name]))
            x,y=set(a["selected_tokens"].tolist()),set(b["selected_tokens"].tolist())
            jaccards.append(len(x&y)/max(len(x|y),1))
        result[str(lid)]={k:{"mean":float(np.mean(v)),"p95":float(np.quantile(v,.95)),"max":max(v)} for k,v in values.items()}
        result[str(lid)]["selected_ids_mean_jaccard"]=float(np.mean(jaccards))
    return result


def audit_pair_geometry(rows, *, store_root,task,source,target,max_requests=8,max_tokens=2048):
    ks,_=read_prefix(store_root,task,source,"key");kt,_=read_prefix(store_root,task,target,"key")
    out=[]
    for r in list(rows.values())[:max_requests]:
        data=r["layers"][target];ids=data["selected_tokens"]
        # Subsample only for SVD audit; report actual diagnostic dimension.
        if len(ids)>max_tokens:ids=ids[torch.linspace(0,len(ids)-1,max_tokens).round().long().unique()]
        hk,d=ks.shape[1:];q=data["q"].float().reshape(-1,hk,data["q"].shape[1]//hk,d)
        head_results=[]
        for h in range(hk):
            z=q[:,h].reshape(-1,d)@kt[ids,h].T*(d**-0.5)
            head_results.append(logit_span_bound(ks[ids,h],z))
        out.append({"query_hash":r["query_hash"],"heads":head_results})
    return {"scope":"unweighted logit-span diagnostic; NOT an accuracy/KL lower bound", "rows":out}


def harm_rate_upper(losses, n, alpha=.05):
    """One-sided exact binomial upper bound; net gain is conservatively >= -harm.
    Does not turn a zero-discordance bootstrap into proof of noninferiority.
    Query independence is required; repeated/clustered queries need grouped design.
    """
    if not 0<=losses<=n or n<1 or not 0<alpha<1:raise ValueError("Bad binomial parameters")
    if losses==n:return 1.0
    if losses==0:return 1-alpha**(1/n)
    def cdf(p):
        if p<=0:return 1.
        if p>=1:return 0.
        lp,lq=math.log(p),math.log1p(-p)
        logs=[math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)+k*lp+(n-k)*lq for k in range(losses+1)]
        mx=max(logs);return math.exp(mx)*sum(math.exp(x-mx) for x in logs)
    lo,hi=losses/n,1.
    for _ in range(64):
        mid=(lo+hi)/2
        if cdf(mid)>alpha:lo=mid
        else:hi=mid
    return (lo+hi)/2


def paired_quality(baseline,student,*,margin_pp=1.0,bootstrap=5000,seed=11):
    """One row/query, paired bootstrap CONDITIONAL ON these fixed checkpoints.

    Reusing this set for adapter selection invalidates a final test claim.
    Correlated queries sharing documents need a suitable grouped protocol.
    """
    def index(rows):
        result={}
        for r in rows:
            key=(str(r.get("task","")),str(r["query_hash"]))
            if key in result:raise ValueError("Duplicate query; aggregate repeated timing separately")
            c=int(r["correct"])
            if c not in (0,1):raise ValueError("correct must be 0 or 1")
            result[key]=c
        return result
    b,s=index(baseline),index(student)
    if set(b)!=set(s) or not b:raise ValueError("Mismatched/empty paired request sets")
    keys=sorted(b);xb=np.array([b[k] for k in keys]);xs=np.array([s[k] for k in keys]);diff=xs-xb
    rng=np.random.default_rng(seed);means=[]
    for _ in range(bootstrap):means.append(float(diff[rng.integers(0,len(diff),len(diff))].mean())*100)
    loss=int(((xb==1)&(xs==0)).sum());gain=int(((xb==0)&(xs==1)).sum());m=loss+gain
    p=1. if m==0 else min(1.,2*sum(math.exp(math.lgamma(m+1)-math.lgamma(k+1)-math.lgamma(m-k+1)-m*math.log(2)) for k in range(min(loss,gain)+1)))
    lo,hi=np.quantile(means,[.025,.975])
    harm_upper=harm_rate_upper(loss,len(keys))
    return {"n":len(keys),"baseline_correct":int(xb.sum()),"student_correct":int(xs.sum()),
        "delta_pp":100*float(diff.mean()),"paired_bootstrap_95pct_delta_pp":[float(lo),float(hi)],
        "baseline_right_student_wrong":loss,"baseline_wrong_student_right":gain,
        "mcnemar_exact_two_sided_p":p,"noninferiority_margin_pp":margin_pp,
        "bootstrap_descriptive_lower_exceeds_margin":bool(lo>=-margin_pp),
        "harm_rate_exact_one_sided_upper95_pp":100*harm_upper,
        "conservative_noninferiority_supported":bool(100*harm_upper<margin_pp),
        "warning":"No multiple-selection or dataset-cluster correction; not proof of generalization"}


def operator_audit(rows, *, store_root, task, source, target, reader, max_requests=32):
    """Same student input, three counterfactuals. Diagnostic only, not task scores."""
    from .refine import distribution
    ks,_=read_prefix(store_root,task,source,"key");vs,_=read_prefix(store_root,task,source,"value")
    kt,_=read_prefix(store_root,task,target,"key");vt,_=read_prefix(store_root,task,target,"value")
    r=reader.to(device="cpu",dtype=torch.float32)
    outputs=[]
    with torch.no_grad():
        for tr,sr in rows[:max_requests]:
            td,sd=tr["layers"][target],sr["layers"][target]
            ids=sd["selected_tokens"]
            q,u,z=[sd[n].float() for n in ("q","tail_output","tail_lse")]
            native,_,_=distribution(q,kt[ids],vt[ids],u,z)
            shared,_,_=distribution(q,ks[ids],vs[ids],u,z,r.A,r.B,r.bk,r.bv)
            k_only,_,_=distribution(q,ks[ids],vt[ids],u,z,r.A,None,r.bk,None)
            v_only,_,_=distribution(q,kt[ids],vs[ids],u,z,None,r.B,None,r.bv)
            teacher=td["attention_output"].float()
            outputs.append({"query_hash":tr["query_hash"],
                "upstream_input_error_native_student_vs_clean_teacher":relative_rmse(native,teacher),
                "local_substitution_shared_vs_native_student":relative_rmse(shared,native),
                "total_shared_vs_clean_teacher":relative_rmse(shared,teacher),
                "k_only_substitution_vs_native_student":relative_rmse(k_only,native),
                "v_only_substitution_vs_native_student":relative_rmse(v_only,native)})
    return {"scope":"local conditional counterfactuals; not whole-model causal proof; relative errors do not add",
            "A_spectral_norm_per_head":torch.linalg.matrix_norm(r.A,ord=2).tolist(),
            "B_spectral_norm_per_head":torch.linalg.matrix_norm(r.B,ord=2).tolist(),"rows":outputs}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",required=True);p.add_argument("--student",required=True)
    p.add_argument("--task",required=True);p.add_argument("--output",required=True)
    p.add_argument("--store-root");p.add_argument("--source",type=int);p.add_argument("--target",type=int)
    p.add_argument("--adapter")
    a=p.parse_args(argv)
    if Path(a.output).exists():raise FileExistsError(a.output)
    t,s=load_traces(a.teacher,a.task),load_traces(a.student,a.task)
    result={"layer_drift":audit_traces(t,s),"scope":"diagnostic; no causal attribution from a single run"}
    if a.store_root is not None:
        if a.source is None or a.target is None:raise ValueError("--source and --target required")
        result["logit_span"]=audit_pair_geometry(t,store_root=a.store_root,task=a.task,source=a.source,target=a.target)
        if a.adapter:
            from contiguous_fuxian.sharedkv.core import Reader
            from contiguous_fuxian.sharedkv.io import load_safe
            bundle=load_safe(a.adapter)
            if bundle["sources"][a.target]!=a.source:raise ValueError("Adapter source mismatch")
            reader=Reader.from_state_dict(bundle["readers"][str(a.target)])
            result["operator_counterfactuals"]=operator_audit(paired(t,s,a.target),store_root=a.store_root,
                task=a.task,source=a.source,target=a.target,reader=reader)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2))

if __name__=="__main__":main()


def mean_label_scores(token_logprobs):
    """Match the original runner: mean continuation log probability per token."""
    if any(not values for values in token_logprobs.values()):
        raise ValueError("Empty label token sequence")
    return {key:sum(values)/len(values) for key,values in token_logprobs.items()}
