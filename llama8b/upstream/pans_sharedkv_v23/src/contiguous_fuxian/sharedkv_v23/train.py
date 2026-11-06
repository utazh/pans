"""One source -> multiple consumers; joint frozen-block reconstruction.

Use freshly captured FULL traces, not the v21 sampled-query traces. MLPs,
normalizations, Q/K/V/O and (optional) final lm_head are frozen but remain in
the autograd graph. No gold labels are loaded. Export remains diagnostic.
"""
from __future__ import annotations
import argparse,copy,json,random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import Group,IDENTITY_FIELDS,BASE,load_dataset,validate_pair,write_new,selected_variants
from .block import TrainableReader,replay_group,boundary_loss


def matched(teacher,student,group):
    if set(teacher)!=set(student):raise ValueError('Teacher/student query sets must match exactly')
    rows=[]
    for key in sorted(teacher):
        validate_pair(teacher[key],student[key],group)
        rows.append((teacher[key],student[key]))
    return rows


def choose_variant(pair,path):
    return tuple(next(v for v in r['variants'] if tuple(v['suffix_ids'])==tuple(path)) for r in pair)


def prepare_example(pair,path,group,all_prefix,device,dtype):
    tr,sr=pair;tv,sv=choose_variant(pair,path)
    h=sv['layer_inputs'][group.start].to(device=device,dtype=dtype)
    th=tv['layer_inputs'][group.start].to(device=device,dtype=dtype)
    targets={lid:y.to(device=device,dtype=dtype) for lid,y in tv['layer_outputs'].items()}
    pos=torch.arange(tr['prefix_tokens'],tr['prefix_tokens']+h.shape[1],device=device)[None]
    prefix={}
    for lid in range(group.start,group.end):
        ids=torch.tensor(sr['selection_by_layer'][lid],device=device,dtype=torch.long)
        prefix[lid]=tuple(x.index_select(0,ids).unsqueeze(0) for x in all_prefix[lid])
    return h,th,targets,pos,prefix,len(tr['query_token_ids']),tv


def token_indices(query_tokens,total_tokens,device):
    return torch.arange(query_tokens-1,total_tokens,device=device)


def loss_for(model,group,inputs,readers,args):
    h,th,targets,pos,prefix,T,tv=inputs
    out,mid,_=replay_group(model,group,h,pos,prefix,readers)
    target=targets[group.end-1]
    loss,_=boundary_loss(out,target,th,T,last_weight=args.last_weight,suffix_weight=args.suffix_weight)
    local=loss*0
    for lid in group.consumers:
        scale=(targets[lid]-tv['layer_inputs'][lid].to(h.device)).float().square().mean().clamp_min(1e-8)
        local=local+(mid[lid].float()-targets[lid].float()).square().mean()/scale
    loss=loss+args.intermediate_weight*local/max(1,len(group.consumers))
    if args.logit_weight:
        if group.end!=len(model.model.layers):raise ValueError('lm_head KL only valid at final group boundary')
        ids=token_indices(T,out.shape[1],out.device)
        student=model.lm_head(model.model.norm(out.index_select(1,ids))).float()
        with torch.no_grad():teacher=model.lm_head(model.model.norm(target.index_select(1,ids))).float()
        logp=teacher.log_softmax(-1)
        kl=(logp.exp()*(logp-student.log_softmax(-1))).sum(-1).mean()
        loss=loss+args.logit_weight*kl
    return loss


def evaluate(model,group,rows,prefix,readers,args,device,dtype):
    losses=[];end_errors=[];identity_errors=[];suffix_errors=[]
    local={lid:{'err':0.,'signal':0.,'max_req':0.,'mass':[]} for lid in group.consumers}
    with torch.no_grad():
        for pair in rows:
            qloss=[];qerr=[];qident=[];qtail=[]
            for v in selected_variants(pair[0],not args.query_only):
                data=prepare_example(pair,v['suffix_ids'],group,prefix,device,dtype)
                h,th,targets,pos,pk,T,tv=data
                out,_,details=replay_group(model,group,h,pos,pk,readers,details=True)
                identity,_,teacher_details=replay_group(model,group,th,pos,pk,readers,native=True,details=True)
                target=targets[group.end-1]
                scale=(target.float()-th.float()).square().mean().clamp_min(1e-8)
                qloss.append(float(loss_for(model,group,data,readers,args)))
                qerr.append(float(((out.float()-target.float()).square().mean()/scale).sqrt()))
                qident.append(float(((identity.float()-target.float()).square().mean()/scale).sqrt()))
                if out.shape[1]>T:qtail.append(float(((out[:,T:].float()-target[:,T:].float()).square().mean()/scale).sqrt()))
                for lid in group.consumers:
                    so,sm=details[lid];_,tm=teacher_details[lid]
                    to=tv['attention_outputs'][lid].to(device).reshape_as(so)
                    e=float((so.float()-to.float()).double().square().sum());s=float(to.double().square().sum())
                    rec=local[lid];rec['err']+=e;rec['signal']+=s
                    rec['max_req']=max(rec['max_req'],(e/max(s,1e-30))**.5)
                    rec['mass'].append((sm-tm).abs().flatten().cpu())
            losses.append(sum(qloss)/len(qloss));end_errors.append(sum(qerr)/len(qerr));identity_errors+=qident
            if qtail:suffix_errors.append(sum(qtail)/len(qtail))
    diagnostics={}
    for lid,rec in local.items():
        rmse=(rec['err']/max(rec['signal'],1e-30))**.5
        p95=float(torch.quantile(torch.cat(rec['mass']),.95))
        diagnostics[str(lid)]={'output_relative_rmse':rmse,'max_request_or_branch_rmse':rec['max_req'],
            'prefix_mass_p95':p95,'old_local_gate_passed':rmse<=.03 and rec['max_req']<=.08 and p95<=.03}
    return {'validation_objective':sum(losses)/len(losses),'queries':len(rows),
        'group_update_relative_rmse':sum(end_errors)/len(end_errors),
        'max_identity_group_replay_relative_rmse':max(identity_errors),
        'suffix_group_update_relative_rmse':sum(suffix_errors)/len(suffix_errors) if suffix_errors else None,
        'local_diagnostics':diagnostics,'end_to_end_validated':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('model-path','store-root','task','teacher-fit','teacher-val','student-fit','student-val','output'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True)
    p.add_argument('--source',type=int,required=True);p.add_argument('--native-layers',default='');p.add_argument('--carry');p.add_argument('--initial-bundle')
    p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=300)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--ridge',type=float,default=1e-3)
    p.add_argument('--eval-every',type=int,default=50);p.add_argument('--seed',type=int,default=23)
    p.add_argument('--last-weight',type=float,default=.5);p.add_argument('--suffix-weight',type=float,default=1.)
    p.add_argument('--intermediate-weight',type=float,default=.1);p.add_argument('--trust-weight',type=float,default=1e-3)
    p.add_argument('--logit-weight',type=float,default=0.);p.add_argument('--query-only',action='store_true')
    p.add_argument('--no-bf16-ste',action='store_true');p.add_argument('--max-sequence',type=int,default=1024)
    p.add_argument('--max-identity-replay-error',type=float,default=.02)
    a=p.parse_args(argv)
    if a.steps<0 or a.eval_every<=0 or min(a.lr,a.ridge)<=0:raise ValueError('Invalid fit parameters')
    if min(a.last_weight,a.suffix_weight,a.intermediate_weight,a.trust_weight,a.logit_weight)<0:raise ValueError('Negative loss weight')
    if Path(a.output).exists():raise FileExistsError(a.output)
    torch.manual_seed(a.seed);rng=random.Random(a.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    tf,tv,sf,sv=[load_dataset(x,a.task) for x in (a.teacher_fit,a.teacher_val,a.student_fit,a.student_val)]
    if (set(tf)|set(sf)) & (set(tv)|set(sv)):raise ValueError('Fit/validation query leakage')
    header=next(iter(tf.values()));g=header['geometry'];group=Group(a.start,a.end,a.source,tuple(int(x) for x in a.native_layers.split(',') if x))
    group.validate(g['layers'],header['period'])
    if not group.consumers:raise ValueError('No consumers requested')
    from contiguous_fuxian.sharedkv.io import load_safe,read_prefix,validate_sources,FORMAT,BASE_COMMIT
    from contiguous_fuxian.sharedkv.core import Reader
    from contiguous_fuxian.sharedkv.fit import affine_ridge
    from contiguous_fuxian.sharedkv_v21.traces import fingerprint
    carry=load_safe(a.carry) if a.carry else None
    sources=list(range(g['layers'])) if carry is None else list(carry['sources'])
    old_readers={} if carry is None else copy.deepcopy(carry['readers'])
    if carry is not None:
        if carry.get('format')!=FORMAT or carry.get('head_maps'):raise ValueError('Convert identity-only v22 bundles explicitly; mapped carry is not supported')
        for k in IDENTITY_FIELDS:
            if carry[k]!=header[k]:raise ValueError(f'Carry identity mismatch: {k}')
    if any(int(k)>=group.start for k in old_readers):raise ValueError('Add groups in ascending order; only earlier frozen groups may be carried')
    for lid in range(group.start,group.end):
        if sources[lid]!=lid or lid in sources[:group.start]:raise ValueError('Group source ownership conflicts with carry')
    fp=fingerprint({int(k):Reader.from_state_dict(v).to(device='cpu',dtype=torch.bfloat16) for k,v in old_readers.items()})
    for datasets,teacher in (((tf,tv),True),((sf,sv),False)):
        for ds in datasets:
            for row in ds.values():
                for k in IDENTITY_FIELDS:
                    if row[k]!=header[k]:raise ValueError(f'Different calibration protocol: {k}')
                expected=list(range(g['layers'])) if teacher else sources
                if row['rollout_sources']!=expected:raise ValueError('Wrong teacher/upstream student source map')
                expected_fp=fingerprint({}) if teacher else fp
                if row['rollout_reader_fingerprint']!=expected_fp:raise ValueError('Stale reader fingerprint')
                for v in selected_variants(row,not a.query_only):
                    if v['layer_inputs'][group.start].shape[1]>a.max_sequence:raise ValueError('Sequence exceeds explicit training memory limit')
    fit=matched(tf,sf,group);val=matched(tv,sv,group)
    from contiguous_fuxian.quantized_key_index import model_identity
    if model_identity(a.model_path)!=header['model_identity']:raise ValueError('Backbone weights/config do not match capture')
    from transformers import AutoModelForCausalLM
    device=torch.device(a.device);dtype=torch.bfloat16 if device.type=='cuda' else torch.float32
    model=AutoModelForCausalLM.from_pretrained(a.model_path,local_files_only=True,dtype=dtype,
        attn_implementation='eager',trust_remote_code=False).to(device).eval()
    if model.config.model_type!='qwen2' or getattr(model.config,'rope_scaling',None) is not None or getattr(model.config,'use_sliding_window',False):
        raise ValueError('Only fixed-RoPE Qwen2/Qwen2.5 supported')
    model.requires_grad_(False)
    prefix_cpu={}
    for lid in range(group.start,group.end):
        kv=[]
        for kind in ('key','value'):
            x,meta=read_prefix(a.store_root,a.task,lid,kind)
            if meta['token_hash']!=header['token_hash']:raise ValueError('Prefix mismatch')
            kv.append(x)
        prefix_cpu[lid]=tuple(kv)
    initial=load_safe(a.initial_bundle) if a.initial_bundle else None
    if initial:
        if initial.get('format')!=FORMAT or initial.get('head_maps'):raise ValueError('Explicitly convert identity-only v22 initialization first')
        for k in IDENTITY_FIELDS:
            if initial[k]!=header[k]:raise ValueError('Initialization identity mismatch')
    modules={}
    for lid in group.consumers:
        if initial and str(lid) in initial['readers'] and initial['sources'][lid]==group.source:
            r=Reader.from_state_dict(initial['readers'][str(lid)])
        else:
            A=[];bk=[];B=[];bv=[]
            for h in range(g['kv_heads']):
                aa,bb=affine_ridge(prefix_cpu[group.source][0][:,h],prefix_cpu[lid][0][:,h],ridge=a.ridge)
                cc,dd=affine_ridge(prefix_cpu[group.source][1][:,h],prefix_cpu[lid][1][:,h],ridge=a.ridge)
                A.append(aa);bk.append(bb);B.append(cc);bv.append(dd)
            r=Reader(torch.stack(A),torch.stack(B),torch.stack(bk),torch.stack(bv))
        modules[str(lid)]=TrainableReader(r,ste=not a.no_bf16_ste)
    readers=nn.ModuleDict(modules).to(device)
    prefix={lid:tuple(x.to(device=device,dtype=dtype) for x in kv) for lid,kv in prefix_cpu.items()}
    del prefix_cpu
    init_params=[p.detach().clone() for p in readers.parameters()]
    best_report=evaluate(model,group,val,prefix,readers,a,device,dtype)
    if best_report['max_identity_group_replay_relative_rmse']>a.max_identity_replay_error:
        raise RuntimeError('Identity replay mismatches captured backend: fix numerics before fitting')
    history=[{'step':0,**best_report}];best=copy.deepcopy(readers.state_dict());best_step=0
    optimizer=torch.optim.Adam(readers.parameters(),lr=a.lr)
    for step in range(1,a.steps+1):
        pair=fit[rng.randrange(len(fit))]
        v=rng.choice(selected_variants(pair[0],not a.query_only))
        data=prepare_example(pair,v['suffix_ids'],group,prefix,device,dtype)
        loss=loss_for(model,group,data,readers,a)
        norm=sum(x.square().sum() for x in init_params).clamp_min(1.)
        trust=sum((p-p0).square().sum() for p,p0 in zip(readers.parameters(),init_params))/norm
        loss=loss+a.trust_weight*trust
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite group loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(p.grad is not None for p in model.parameters()):raise RuntimeError('Backbone received parameter gradients')
        torch.nn.utils.clip_grad_norm_(readers.parameters(),1.);optimizer.step()
        if step%a.eval_every==0 or step==a.steps:
            report=evaluate(model,group,val,prefix,readers,a,device,dtype)
            history.append({'step':step,**report})
            if report['validation_objective']<best_report['validation_objective']:
                best_report=report;best=copy.deepcopy(readers.state_dict());best_step=step
    readers.load_state_dict(best)
    # Report the exported BF16-rounded reader, including --no-bf16-ste ablations.
    with torch.no_grad():
        for parameter in readers.parameters():parameter.copy_(parameter.to(torch.bfloat16).float())
    best_report=evaluate(model,group,val,prefix,readers,a,device,dtype)
    proposed=sources.copy()
    for lid in group.consumers:
        proposed[lid]=group.source;old_readers[str(lid)]=readers[str(lid)].export().state_dict()
    validate_sources(proposed,period=header['period'])
    bans=set(tf)|set(tv)
    for bundle in (carry,initial):
        if bundle:bans.update(bundle['calibration_query_hashes'])
    bundle={'format':FORMAT,'base_commit':BASE_COMMIT,**{k:header[k] for k in IDENTITY_FIELDS},
        'sources':proposed,'readers':old_readers,'calibration_query_hashes':sorted(bans),
        'diagnostic_only':True,'deployment_status':'group_candidate_not_accepted',
        'accepted_consumers':0,'candidate_consumers':len(old_readers),'end_to_end_validated':False,
        'compiler_family':'v23_group_boundary','audited_runtime_base':BASE,
        'group':{'start':group.start,'end':group.end,'source':group.source,'native':list(group.native)},
        'best_step':best_step,'report':best_report,'compiler_args':vars(a),
        'local_gate_passed':all(x['old_local_gate_passed'] for x in best_report['local_diagnostics'].values())}
    write_new(a.output,bundle)
    Path(a.output+'.report.json').write_text(json.dumps({'history':history,'best_step':best_step,
        'report':best_report,'args':vars(a),'unique_sources':len(set(proposed))},indent=2,allow_nan=False))
    print(json.dumps({'output':a.output,'unique_sources':len(set(proposed)),
        'best_step':best_step,'report':best_report},indent=2))

if __name__=='__main__':main()
