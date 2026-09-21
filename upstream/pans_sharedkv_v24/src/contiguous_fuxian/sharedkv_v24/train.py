"""Output-score-aligned multi-group reader training; no backbone optimization.

Fits a requested source map, rather than merging independently calibrated
checkpoints. Output remains a diagnostic candidate; training is not approval.
"""
from __future__ import annotations
import argparse,copy,json,random,time
from pathlib import Path
import torch
from torch import nn
from contiguous_fuxian.sharedkv.core import Reader
from contiguous_fuxian.sharedkv.io import load_safe,read_prefix,FORMAT as BUNDLE_FORMAT,BASE_COMMIT
from contiguous_fuxian.sharedkv.fit import affine_ridge
from contiguous_fuxian.sharedkv_v23.block import TrainableReader
from .records import FIELDS,PLAN_FORMAT,load,assert_disjoint,assert_plan,validate_sources,save_new,json_digest
from .replay import selected_bank,continuation_tokens,objective,score_vectors,streamed_backward


def settings(args):
    return dict(temperature=args.temperature,score_weight=args.score_weight,token_weight=args.token_weight,
                rank_weight=args.rank_weight,margin_cap=args.margin_cap)


def teacher_vectors(row,device):
    return [torch.tensor(row['teacher_token_logprobs'][label],device=device,dtype=torch.float32)
            for label in row['label_order']]


def initialize(raw,sources,initial,geometry,ridge,device):
    result={};inherited={}
    if initial:
        if initial.get('format')!=BUNDLE_FORMAT or initial.get('head_maps'):
            raise ValueError('Only standard same-number-head bundles supported; no silent conversion')
        inherited=initial['readers']
        for lid,source in enumerate(initial['sources']):
            if lid!=source and sources[lid]!=source:
                raise ValueError('Expansion must preserve existing ownership; use separate explicitly reinitialized experiments')
    for lid,source in enumerate(sources):
        if source==lid:continue
        if str(lid) in inherited:
            r=Reader.from_state_dict(inherited[str(lid)])
        else:
            A=[];B=[];bk=[];bv=[]
            for head in range(geometry['kv_heads']):
                aa,bb=affine_ridge(raw[source][0][:,head].float(),raw[lid][0][:,head].float(),ridge=ridge)
                cc,dd=affine_ridge(raw[source][1][:,head].float(),raw[lid][1][:,head].float(),ridge=ridge)
                A.append(aa);bk.append(bb);B.append(cc);bv.append(dd)
            r=Reader(torch.stack(A),torch.stack(B),torch.stack(bk),torch.stack(bv))
        result[str(lid)]=TrainableReader(r,ste=True)
    return nn.ModuleDict(result).to(device)


def evaluate(model,rows,raw,sources,readers,args,*,native_parity=False):
    device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    sums={};flips=0;max_score=0.;max_token=0.;requests=[]
    native=list(range(len(sources)))
    with torch.no_grad():
        for key,row in sorted(rows.items()):
            active=native if native_parity else sources
            bank=selected_bank(raw,row,active,device,dtype)
            rr={} if native_parity else readers
            got=[continuation_tokens(model,row,label,active,bank,rr,checkpoint_layers=False,max_sequence=args.max_sequence)
                 for label in row['label_order']]
            target=teacher_vectors(row,device)
            loss,parts=objective(got,target,**settings(args))
            gs,ts=score_vectors(got),score_vectors(target)
            err=float((gs-ts).abs().max());te=max(float((a-b).abs().max()) for a,b in zip(got,target))
            centered_error=float(((gs-gs.mean())-(ts-ts.mean())).abs().max())
            ordered=torch.sort(ts,descending=True).values
            margin=float(ordered[0]-ordered[1])
            flips+=int(int(gs.argmax())!=int(ts.argmax()))
            max_score=max(max_score,err);max_token=max(max_token,te)
            vals={'objective':float(loss),**{k:float(v) for k,v in parts.items()}}
            for k,v in vals.items():sums[k]=sums.get(k,0.)+v
            requests.append({'query_hash':key,'max_score_error':err,'max_token_logprob_error':te,
                             'teacher_top':int(ts.argmax()),'student_top':int(gs.argmax()),
                             'teacher_margin':margin,'max_centered_score_error':centered_error,
                             'sufficient_teacher_decision_preservation_bound':margin>2*centered_error})
            del bank
    return {**{k:v/len(rows) for k,v in sums.items()},'requests':len(rows),'teacher_decision_flips':flips,
            'max_abs_mean_score_error':max_score,'max_abs_token_logprob_error':max_token,
            'query_diagnostics':requests,'gold_accuracy_evaluated':False,
            'decision_margin_bound_fraction':sum(r['sufficient_teacher_decision_preservation_bound'] for r in requests)/len(requests)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model-path','store-root','task','teacher-fit','teacher-val','plan','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--initial');p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=200)
    p.add_argument('--seed',type=int,default=24);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--ridge',type=float,default=1e-3);p.add_argument('--trust-weight',type=float,default=1e-3)
    p.add_argument('--eval-every',type=int,default=25);p.add_argument('--max-sequence',type=int,default=256)
    p.add_argument('--temperature',type=float,default=1.);p.add_argument('--score-weight',type=float,default=1.)
    p.add_argument('--token-weight',type=float,default=.1);p.add_argument('--rank-weight',type=float,default=.25)
    p.add_argument('--margin-cap',type=float,default=.5);p.add_argument('--no-checkpoint',action='store_true')
    p.add_argument('--max-native-score-error',type=float,default=.02)
    p.add_argument('--max-native-token-error',type=float,default=.10)
    a=p.parse_args()
    if a.steps<0 or min(a.lr,a.ridge,a.eval_every,a.max_sequence,a.temperature)<=0:
        raise ValueError('Invalid optimization settings')
    if min(a.trust_weight,a.score_weight,a.token_weight,a.rank_weight,a.margin_cap)<0:
        raise ValueError('Negative loss weights')
    out=Path(a.output)
    if out.exists() or Path(str(out)+'.report.json').exists():raise FileExistsError(out)
    fit,val=load(a.teacher_fit,a.task,'fit'),load(a.teacher_val,a.task,'validation');assert_disjoint(fit,val)
    header=next(iter(fit.values()));plan=json.loads(Path(a.plan).read_text())
    if plan.get('format')!=PLAN_FORMAT:raise ValueError('Not a v24 topology plan')
    for key in FIELDS:
        if plan[key]!=header[key]:raise ValueError(f'Plan identity mismatch: {key}')
    if not set(plan['design_query_hashes'])<=set(fit):raise ValueError('Topology was not designed exclusively on the fit set')
    sources=plan['sources'];validate_sources(sources,header['period'])
    for row in list(fit.values())+list(val.values()):assert_plan(row,sources)
    if sources==list(range(len(sources))):raise ValueError('No reader to train')
    initial=load_safe(a.initial) if a.initial else None
    if initial:
        for key in FIELDS:
            if initial[key]!=header[key]:raise ValueError(f'Initialization mismatch: {key}')
        # The local validation split must not become a fit split through a prior stage.
        prior_fit=set(initial.get('fit_query_hashes',[]));prior_val=set(initial.get('validation_query_hashes',[]))
        if prior_fit&set(val) or prior_val&set(fit):raise ValueError('Initial bundle swaps known fit/validation roles')
        if not prior_fit and not prior_val and set(initial['calibration_query_hashes']) & (set(fit)|set(val)):
            raise ValueError('Legacy bundle has no fit/val role provenance. Use disjoint new calibration queries or omit --initial; do not guess split roles.')
    torch.manual_seed(a.seed);rng=random.Random(a.seed);torch.set_num_threads(min(4,torch.get_num_threads()))
    from contiguous_fuxian.quantized_key_index import model_identity
    if model_identity(a.model_path)!=header['model_identity']:raise ValueError('Model/adapter/config fingerprint mismatch')
    from transformers import AutoModelForCausalLM
    device=torch.device(a.device);dtype=torch.bfloat16 if device.type=='cuda' else torch.float32
    model=AutoModelForCausalLM.from_pretrained(a.model_path,local_files_only=True,dtype=dtype,
        attn_implementation='eager',trust_remote_code=False).to(device).eval()
    if model.config.model_type!='qwen2' or getattr(model.config,'rope_scaling',None) is not None or getattr(model.config,'use_sliding_window',False):
        raise ValueError('Only fixed-RoPE Qwen2 full attention supported')
    model.requires_grad_(False)
    raw_cpu={}
    for lid in range(len(sources)):
        values=[]
        for kind in ('key','value'):
            x,meta=read_prefix(a.store_root,a.task,lid,kind)
            if meta['token_hash']!=header['token_hash']:raise ValueError('Prefix identity mismatch')
            values.append(x)
        raw_cpu[lid]=tuple(values)
    readers=initialize(raw_cpu,sources,initial,header['geometry'],a.ridge,device)
    raw={lid:tuple(x.to(device=device,dtype=dtype) for x in kv) for lid,kv in raw_cpu.items()};del raw_cpu
    out.parent.mkdir(parents=True,exist_ok=True)
    parity=evaluate(model,val,raw,sources,readers,a,native_parity=True)
    if parity['max_abs_mean_score_error']>a.max_native_score_error or parity['max_abs_token_logprob_error']>a.max_native_token_error:
        Path(str(out)+'.rejected_replay.json').write_text(json.dumps(parity,indent=2))
        raise RuntimeError('Native score replay exceeds predeclared tolerances. Fix numerical/backend mismatch before training.')
    initial_parameters=[x.detach().clone() for x in readers.parameters()]
    initial_report=evaluate(model,val,raw,sources,readers,a)
    best=copy.deepcopy(readers.state_dict());best_report=initial_report;best_step=0
    history=[{'step':0,**{k:v for k,v in initial_report.items() if k!='query_diagnostics'}}]
    optimizer=torch.optim.Adam(readers.parameters(),lr=a.lr)
    rows=list(fit.values());branch_forwards=0;backward_branches=0
    started=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for step in range(1,a.steps+1):
        row=rows[rng.randrange(len(rows))];bank=selected_bank(raw,row,sources,device,dtype)
        labels=row['label_order'];target=teacher_vectors(row,device)
        def compute(label):
            return continuation_tokens(model,row,label,sources,bank,readers,
                checkpoint_layers=not a.no_checkpoint,max_sequence=a.max_sequence)
        optimizer.zero_grad(set_to_none=True)
        loss,_,_=streamed_backward(compute,labels,target,settings(a))
        branch_forwards+=2*len(labels);backward_branches+=len(labels)
        norm=sum(x.square().sum() for x in initial_parameters).clamp_min(1.)
        trust=sum((p-p0).square().sum() for p,p0 in zip(readers.parameters(),initial_parameters))/norm
        if a.trust_weight:(a.trust_weight*trust).backward()
        if any(x.grad is not None for x in model.parameters()):raise RuntimeError('Backbone received gradients')
        if not all(x.grad is None or bool(torch.isfinite(x.grad).all()) for x in readers.parameters()):
            raise FloatingPointError('Reader gradients are nonfinite')
        torch.nn.utils.clip_grad_norm_(readers.parameters(),1.);optimizer.step();del bank
        if step%a.eval_every==0 or step==a.steps:
            report=evaluate(model,val,raw,sources,readers,a)
            history.append({'step':step,**{k:v for k,v in report.items() if k!='query_diagnostics'}})
            print(json.dumps(history[-1]),flush=True)
            if report['objective']<best_report['objective']:
                best=copy.deepcopy(readers.state_dict());best_report=report;best_step=step
    readers.load_state_dict(best)
    with torch.no_grad():
        for parameter in readers.parameters():parameter.copy_(parameter.to(torch.bfloat16).float())
    final=evaluate(model,val,raw,sources,readers,a)
    if device.type=='cuda':torch.cuda.synchronize()
    fit_seconds=time.perf_counter()-started
    bans=set(fit)|set(val)|set(plan['design_query_hashes'])
    if initial:bans.update(initial['calibration_query_hashes'])
    bundle={'format':BUNDLE_FORMAT,'base_commit':BASE_COMMIT,**{k:header[k] for k in FIELDS},
        'sources':sources,'readers':{k:r.export().state_dict() for k,r in readers.items()},
        'calibration_query_hashes':sorted(bans),'fit_query_hashes':sorted(set(fit)|set(initial.get('fit_query_hashes',[]) if initial else [])),
        'validation_query_hashes':sorted(set(val)|set(initial.get('validation_query_hashes',[]) if initial else [])),
        'diagnostic_only':True,'deployment_status':'score_candidate_not_accepted','accepted_consumers':0,
        'candidate_consumers':len(readers),'end_to_end_validated':False,'local_gate_passed':False,
        'old_local_gate_status':'not_recomputed_by_score_trainer_do_not_claim_pass',
        'compiler_family':'v24_score_aligned_source_budget','best_step':best_step,'report':final,
        'compiler_args':vars(a),'topology_digest':json_digest(plan),
        'parent_report':initial.get('report') if initial else None,
        'scope':'Task-conditioned candidates; all-layer student replay; fixed teacher selection at training; online selector at evaluation'}
    save_new(out,bundle)
    report={'args':vars(a),'parity':parity,'history':history,'best_step':best_step,'final':final,
        'unique_sources':len(set(sources)),'training_seconds_including_validation':fit_seconds,
        'explicit_training_branch_forwards':branch_forwards,'explicit_training_backward_branches':backward_branches,
        'activation_checkpoint_recomputation_included_in_walltime_not_forward_count':not a.no_checkpoint,
        'peak_cuda_allocated':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
        'note':'No deployment or noninferiority certification. Compare train compute as well as optimizer steps.'}
    Path(str(out)+'.report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'output':str(out),'unique_sources':len(set(sources)),'best_step':best_step}),flush=True)

if __name__=='__main__':main()
