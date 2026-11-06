"""Paired asymmetric reader calibration; optional adapter-only distillation.

Teacher: clean baseline rollout. Student inputs: actual upstream shared rollout.
Only A/B/bk/bv are optimized, NEVER backbone weights. Prefix/tail normalizers
are jointly preserved by including the whole tail as one extra probability bin.
This is a local surrogate, NOT block reconstruction or end-to-end distillation.
"""
from __future__ import annotations
import argparse,copy,json,random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from contiguous_fuxian.sharedkv.core import Reader
from contiguous_fuxian.sharedkv.fit import fit_key_reader,affine_ridge,ridge_design
from contiguous_fuxian.sharedkv.io import read_prefix,save_new,load_safe,FORMAT,BASE_COMMIT,validate_sources
from .traces import load_traces,paired,assert_splits,IDENTITY_FIELDS,fingerprint


def distribution(q, k, v, tail_out, tail_lse, A=None, B=None, bk=None, bv=None):
    """All inputs float32/64. q [T,Hq,D], K/V [N,Hkv,D]. Differentiable."""
    t,hq,d=q.shape; n,hk,_=k.shape
    if hq%hk or n==0:raise ValueError("Bad GQA/empty prefix")
    groups=hq//hk
    x=q.reshape(t,hk,groups,d)
    if A is not None:x=torch.einsum("thgj,hij->thgi",x,A)
    scores=torch.einsum("thgd,nhd->hgtn",x,k).reshape(hq,t,n)*(d**-0.5)
    if bk is not None:
        offset=torch.einsum("thgd,hd->thg",q.reshape(t,hk,groups,d),bk)
        scores=scores+offset.reshape(t,hq).T[...,None]*(d**-0.5)
    logp=torch.cat((scores,tail_lse.T[...,None]),dim=-1).log_softmax(-1)
    probs=logp.exp()
    ppre=probs[...,:n].reshape(hk,groups,t,n)
    u=torch.einsum("hgtn,nhd->thgd",ppre,v)
    beta=1-probs[...,-1].T
    if B is not None:u=torch.einsum("thgi,hij->thgj",u,B)
    if bv is not None:u=u+beta.reshape(t,hk,groups)[...,None]*bv[None,:,None,:]
    out=u.reshape(t,hq,d)+(1-beta)[...,None]*tail_out
    return out,logp,beta


class TrainableReader(nn.Module):
    def __init__(self,reader,*,ste=False):
        super().__init__();self.ste=ste
        for name in ("A","B","bk","bv"):
            self.register_parameter(name,nn.Parameter(getattr(reader,name).detach().clone().float()))
    def tensors(self):
        values=[getattr(self,n) for n in ("A","B","bk","bv")]
        if self.ste:
            values=[v+(v.to(torch.bfloat16).float()-v).detach() for v in values]
        return values
    def export(self):
        return Reader(*(getattr(self,n).detach().to(torch.bfloat16).float().cpu() for n in ("A","B","bk","bv")))


def objective(student_out, student_logp, teacher_out, teacher_logp, *, student_hidden=None,
              teacher_pre_mlp=None, teacher_hidden=None, o_weight=None):
    att=(student_out-teacher_out).square().mean()/teacher_out.square().mean().clamp_min(1e-8)
    kl=(teacher_logp.exp()*(teacher_logp-student_logp)).sum(-1).mean()
    post=att.new_zeros(())
    if o_weight is not None:
        pred=student_hidden+F.linear(student_out.flatten(1),o_weight)
        # Normalize to the teacher attention UPDATE, not the much larger residual.
        scale=(teacher_pre_mlp-teacher_hidden).square().mean().clamp_min(1e-8)
        post=(pred-teacher_pre_mlp).square().mean()/scale
    return att,kl,post


def tensor_data(record,target,device):
    return {k:(v.to(device=device,dtype=torch.float32) if k not in ("selected_tokens","query_positions")
               else v.to(device)) for k,v in record["layers"][target].items()}


def sample_outputs(row,target,ks,vs,kt,vt,params,device):
    tr,sr=row
    t,s=tensor_data(tr,target,device),tensor_data(sr,target,device)
    ids=s["selected_tokens"]
    with torch.no_grad():
        _,tp,tb=distribution(t["q"],kt[ids],vt[ids],t["tail_output"],t["tail_lse"])
        teacher_out=t["attention_output"]  # actual numerical teacher path, not a fabricated target
    so,sp,sb=distribution(s["q"],ks[ids],vs[ids],s["tail_output"],s["tail_lse"],*params)
    return so,sp,sb,teacher_out,tp,tb,t,s


def evaluate(rows,target,ks,vs,kt,vt,reader,device,o_weight):
    params=[getattr(reader,n).to(device) for n in ("A","B","bk","bv")]
    err=sig=0.;req=[];mass=[];kls=[];post=[]
    with torch.no_grad():
        for row in rows:
            so,sp,sb,to,tp,tb,t,s=sample_outputs(row,target,ks,vs,kt,vt,params,device)
            e=(so-to).double().square().sum().item(); z=to.double().square().sum().item()
            err+=e;sig+=z;req.append((e/max(z,1e-30))**.5)
            mass.append((sb-tb).abs().flatten().cpu());kls.append(float((tp.exp()*(tp-sp)).sum(-1).mean()))
            _,_,po=objective(so,sp,to,tp,student_hidden=s["hidden_in"],teacher_pre_mlp=t["pre_mlp"],
                teacher_hidden=t["hidden_in"],o_weight=o_weight)
            post.append(float(po)**.5)
    return {"output_relative_rmse":(err/max(sig,1e-30))**.5,"max_request_output_relative_rmse":max(req),
        "prefix_mass_p95":float(torch.quantile(torch.cat(mass),.95)),"mean_joint_attention_kl":sum(kls)/len(kls),
        "mean_post_projection_update_relative_rmse":sum(post)/len(post),"requests":len(rows)}


def fit_value_asymmetric(rows,target,ks,vs,kt,vt,reader,*,ridge,device):
    """No-gradient alternative: fix A, fit B on student inputs to clean teacher output."""
    hk,d,_=reader.B.shape
    designs=[[] for _ in range(hk)];targets=[[] for _ in range(hk)]
    identity=Reader(reader.A,torch.eye(d).repeat(hk,1,1),reader.bk,torch.zeros(hk,d))
    params=[getattr(identity,n).to(device) for n in ("A","B","bk","bv")]
    with torch.no_grad():
        for row in rows:
            so,_,bs,to,_,_,_,s=sample_outputs(row,target,ks,vs,kt,vt,params,device)
            private=(1-bs)[...,None]*s["tail_output"]
            up=so-private  # beta_s times conditional source output
            target_out=to-private
            groups=so.shape[1]//hk
            for h in range(hk):
                sl=slice(h*groups,(h+1)*groups)
                x=torch.cat((up[:,sl].reshape(-1,d),bs[:,sl].reshape(-1,1)),-1)
                designs[h].append(x.cpu());targets[h].append(target_out[:,sl].reshape(-1,d).cpu())
    B=[];bv=[]
    for h in range(hk):
        prior=torch.cat((reader.B[h],reader.bv[h][None]),0)
        b,c=ridge_design(torch.cat(designs[h]),torch.cat(targets[h]),ridge=ridge,prior=prior)
        B.append(b);bv.append(c)
    return Reader(reader.A,torch.stack(B),reader.bk,torch.stack(bv))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("store-root","task","teacher-fit","teacher-val","student-fit","student-val","output"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--source",type=int,required=True);p.add_argument("--target",type=int,required=True)
    p.add_argument("--initial-reader",help="Published v1 candidate initialization; not an upstream carry")
    p.add_argument("--carry",help="Exact upstream bundle used to collect student traces; omit at stage 1")
    p.add_argument("--device",default="cpu");p.add_argument("--steps",type=int,default=0)
    p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--ridge",type=float,default=1e-3)
    p.add_argument("--seed",type=int,default=17);p.add_argument("--tokens",type=int,default=8192)
    p.add_argument("--key-mode",choices=["dense","direct","rope_rr"],default="dense")
    p.add_argument("--rank",type=int,default=16);p.add_argument("--eval-every",type=int,default=20)
    p.add_argument("--kl-weight",type=float,default=1.0);p.add_argument("--post-weight",type=float,default=1.0)
    p.add_argument("--trust-weight",type=float,default=1e-3);p.add_argument("--bf16-ste",action="store_true")
    p.add_argument("--max-output-error",type=float,default=.03)
    p.add_argument("--max-request-output-error",type=float,default=.08)
    p.add_argument("--max-mass-p95",type=float,default=.03)
    a=p.parse_args(argv)
    if a.steps<0 or min(a.lr,a.ridge,a.tokens,a.eval_every)<=0:raise ValueError("Invalid optimization settings")
    if min(a.kl_weight,a.post_weight,a.trust_weight)<0:raise ValueError("Loss weights must be nonnegative")
    if Path(a.output).exists():raise FileExistsError(a.output)
    torch.manual_seed(a.seed);rng=random.Random(a.seed);torch.set_num_threads(min(torch.get_num_threads(),4))
    tf,tv,sf,sv=[load_traces(x,a.task) for x in (a.teacher_fit,a.teacher_val,a.student_fit,a.student_val)]
    assert_splits(tf,tv);assert_splits(sf,sv)
    fit=paired(tf,sf,a.target);val=paired(tv,sv,a.target)
    header=next(iter(tf.values()));g=header["geometry"]
    for dataset in (tf,tv,sf,sv):
        for row in dataset.values():
            for f in IDENTITY_FIELDS:
                if row[f]!=header[f]:raise ValueError(f"Different calibration configurations: {f}")
    for dataset in (tf,tv):
        if next(iter(dataset.values()))["rollout_sources"]!=list(range(g["layers"])):
            raise ValueError("Teacher must be the clean baseline, not the previous student")
    sh=next(iter(sf.values()));vh=next(iter(sv.values()))
    for f in ("rollout_sources","rollout_reader_fingerprint"):
        if sh[f]!=vh[f]:raise ValueError("Fit/val students use different upstream readers")
    carry=load_safe(a.carry) if a.carry else None
    sources=list(range(g["layers"])) if carry is None else list(carry["sources"])
    readers={} if carry is None else copy.deepcopy(carry["readers"])
    if sh["rollout_sources"]!=sources:raise ValueError("Student traces not collected from --carry source map")
    carried_readers={int(l):Reader.from_state_dict(r).to(device="cpu",dtype=torch.bfloat16) for l,r in readers.items()}
    if sh["rollout_reader_fingerprint"]!=fingerprint(carried_readers):
        raise ValueError("Student traces are stale: upstream reader fingerprint mismatch")
    if carry is not None:
        for f in IDENTITY_FIELDS:
            if carry[f]!=header[f]:raise ValueError(f"Carry mismatch: {f}")
    if any(int(l)>=a.target for l in readers):
        raise ValueError("Sequential v2 adds consumers in increasing target-layer order")
    if sources[a.target]!=a.target or sources[a.source]!=a.source or a.target==a.source:
        raise ValueError("Consumer must be new; source must own its physical KV")
    if a.target in sources[:a.target]:raise ValueError("Target is already an upstream source")
    proposed=sources.copy();proposed[a.target]=a.source;validate_sources(proposed,period=header["period"])
    for row in list(sf.values())+list(sv.values()):
        plan=row["selection_by_layer"]
        if plan[a.source]!=plan[a.target]:raise ValueError("Source/consumer selections differ")
    ks,meta=read_prefix(a.store_root,a.task,a.source,"key")
    vs,_=read_prefix(a.store_root,a.task,a.source,"value")
    kt,_=read_prefix(a.store_root,a.task,a.target,"key")
    vt,_=read_prefix(a.store_root,a.task,a.target,"value")
    if meta["token_hash"]!=header["token_hash"]:raise ValueError("Wrong prefix")
    n,h,d=ks.shape;ids=torch.linspace(0,n-1,min(n,a.tokens)).round().long().unique()
    q=torch.cat([r["layers"][a.target]["q"].float() for r in sf.values()]).reshape(-1,h,g["q_heads"]//h,d)
    AA=[];bb=[];BB=[];cc=[]
    for head in range(h):
        if a.key_mode=="direct":A,b=torch.eye(d),torch.zeros(d)
        else:A,b,_=fit_key_reader(ks[ids,head],kt[ids,head],q[:,head].reshape(-1,d),
                                  rank=min(a.rank,d),ridge=a.ridge,mode=a.key_mode)
        B,c=affine_ridge(vs[ids,head],vt[ids,head],ridge=a.ridge)
        AA.append(A);bb.append(b);BB.append(B);cc.append(c)
    reader=Reader(torch.stack(AA),torch.stack(BB),torch.stack(bb),torch.stack(cc))
    init_bundle=None
    if a.initial_reader:
        init_bundle=load_safe(a.initial_reader)
        for f in IDENTITY_FIELDS:
            if init_bundle[f]!=header[f]:raise ValueError(f"Initial reader identity mismatch: {f}")
        if init_bundle["sources"][a.target]!=a.source:raise ValueError("Initial reader source mismatch")
        reader=Reader.from_state_dict(init_bundle["readers"][str(a.target)]).to(device="cpu",dtype=torch.float32)
    device=torch.device(a.device)
    ks,vs,kt,vt=[x.to(device) for x in (ks,vs,kt,vt)]
    weights=load_safe(Path(a.teacher_fit)/"weights"/f"layer_{a.target:02d}.pt")
    if weights["model_identity"]!=header["model_identity"]:raise ValueError("o_proj identity mismatch")
    wo=weights["o_proj"].float().to(device).detach()
    reader=fit_value_asymmetric(fit,a.target,ks,vs,kt,vt,reader,ridge=a.ridge,device=device)
    module=TrainableReader(reader,ste=a.bf16_ste).to(device)
    initial=[x.detach().clone() for x in module.parameters()]
    best=module.export();initial_report=evaluate(val,a.target,ks,vs,kt,vt,best,device,wo)
    def score(r):return r["output_relative_rmse"]**2+a.kl_weight*r["mean_joint_attention_kl"]+a.post_weight*r["mean_post_projection_update_relative_rmse"]**2
    best_score=score(initial_report);history=[{"step":0,**initial_report}]
    optimizer=torch.optim.Adam(module.parameters(),lr=a.lr)
    for step in range(1,a.steps+1):
        row=fit[rng.randrange(len(fit))]
        so,sp,_,to,tp,_,t,s=sample_outputs(row,a.target,ks,vs,kt,vt,module.tensors(),device)
        att,kl,post=objective(so,sp,to,tp,student_hidden=s["hidden_in"],teacher_pre_mlp=t["pre_mlp"],
                             teacher_hidden=t["hidden_in"],o_weight=wo)
        trust=sum((x-x0).square().sum() for x,x0 in zip(module.parameters(),initial))/sum(x0.square().sum() for x0 in initial).clamp_min(1.)
        loss=att+a.kl_weight*kl+a.post_weight*post+a.trust_weight*trust
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite calibration loss")
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(module.parameters(),1.)
        optimizer.step()
        if step%a.eval_every==0 or step==a.steps:
            export=module.export();r=evaluate(val,a.target,ks,vs,kt,vt,export,device,wo)
            history.append({"step":step,**r})
            if score(r)<best_score:best,best_score=export,score(r)
    report=evaluate(val,a.target,ks,vs,kt,vt,best,device,wo)
    local_pass=(report["output_relative_rmse"]<=a.max_output_error and
        report["max_request_output_relative_rmse"]<=a.max_request_output_error and
        report["prefix_mass_p95"]<=a.max_mass_p95)
    readers[str(a.target)]=best.state_dict()
    bans=set(tf)|set(tv)|(set(carry["calibration_query_hashes"]) if carry else set())
    if init_bundle is not None:bans.update(init_bundle["calibration_query_hashes"])
    bundle={"diagnostic_only":True,"format":FORMAT,"base_commit":BASE_COMMIT,**{f:header[f] for f in IDENTITY_FIELDS},
        "sources":proposed,"readers":readers,"calibration_query_hashes":sorted(bans),
        "accepted_consumers":0,"candidate_consumers":len(readers),
        "deployment_status":"diagnostic_candidate_not_accepted","end_to_end_validated":False,
        "local_gate_passed":local_pass,"last_consumer":a.target,"report":report,
        "compiler_args":vars(a),"calibration_rollout_fingerprint":sh["rollout_reader_fingerprint"],
        "warning":"Local asymmetric calibration is not whole-model validation; failed candidates never auto-promote"}
    save_new(a.output,bundle)
    Path(a.output).with_suffix(".report.json").write_text(json.dumps({"initial":initial_report,"final":report,
        "local_gate_passed":local_pass,"history":history,"args":vars(a),"end_to_end_validated":False},indent=2))
    print(json.dumps({"output":a.output,"local_gate_passed":local_pass,**report},indent=2))

if __name__=="__main__":main()
