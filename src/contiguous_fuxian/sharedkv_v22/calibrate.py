"""GQA head assignment + the existing v2.1 asymmetric reader calibration.

Fit queries are internally split for fitting edge readers vs selecting a head
bijection. External val remains for checkpoint selection/local diagnostics.
No test request, label, new KV basis, or backbone update is used.
"""
from __future__ import annotations
import argparse
import copy
import json
import random
from pathlib import Path
import torch
from contiguous_fuxian.sharedkv.core import Reader
from contiguous_fuxian.sharedkv.fit import fit_key_reader, affine_ridge, ridge_design
from contiguous_fuxian.sharedkv.io import read_prefix, load_safe, save_new, BASE_COMMIT, validate_sources
from contiguous_fuxian.sharedkv_v21 import refine as old
from contiguous_fuxian.sharedkv_v21.traces import load_traces, paired, assert_splits, IDENTITY_FIELDS
from .matching import FORMAT, BASE_V2, choose_assignment, split_fit, normalized_maps, aligned_fingerprint


def _group(record, target, head, heads, device):
    r = record['layers'][target]
    group = r['q'].shape[1] // heads
    sl = slice(head*group,(head+1)*group)
    return {key: r[key][:,sl].float().to(device) for key in
            ('q','tail_output','tail_lse','attention_output')}


def fit_edge(rows, target, source_head, target_head, ks, vs, kt, vt, *, ridge, device,
             key_tokens=8192, key_mode='dense', rank=16, incumbent=None):
    """Fit ONE target GQA group from ONE source physical K/V head."""
    n,heads,d = ks.shape
    ids = torch.linspace(0,n-1,min(n,key_tokens)).round().long().unique().to(ks.device)
    qrows = torch.cat([_group(s,target,target_head,heads,device)['q'].reshape(-1,d)
                       for _,s in rows])
    if incumbent is None:
        a,bk,_ = fit_key_reader(ks[ids,source_head],kt[ids,target_head],qrows,
                                rank=min(rank,d),ridge=ridge,mode=key_mode)
        b,bv = affine_ridge(vs[ids,source_head],vt[ids,target_head],ridge=ridge)
    else:
        a,b,bk,bv = [getattr(incumbent,k)[target_head].cpu().float()
                      for k in ('A','B','bk','bv')]
    designs,targets = [],[]
    eye = torch.eye(d,device=device)[None]
    zero = torch.zeros(1,d,device=device)
    with torch.no_grad():
        for tr,sr in rows:
            s = _group(sr,target,target_head,heads,device)
            t = _group(tr,target,target_head,heads,device)
            selected = sr['layers'][target]['selected_tokens'].to(device)
            out,_,beta = old.distribution(s['q'],ks[selected,source_head:source_head+1],
                vs[selected,source_head:source_head+1],s['tail_output'],s['tail_lse'],
                a.to(device)[None],eye,bk.to(device)[None],zero)
            private = (1-beta)[...,None]*s['tail_output']
            prefix = out-private
            designs.append(torch.cat((prefix.reshape(-1,d),beta.reshape(-1,1)),-1).cpu())
            targets.append((t['attention_output']-private).reshape(-1,d).cpu())
    b,bv = ridge_design(torch.cat(designs),torch.cat(targets),ridge=ridge,
                        prior=torch.cat((b,bv[None]),0))
    # Evaluate assignment with the actual BF16 parameter export, not ideal FP64 parameters.
    return Reader(*(x[None].to(torch.bfloat16).float() for x in (a,b,bk,bv)))


def edge_report(rows,target,source_head,target_head,ks,vs,kt,vt,reader,device):
    heads = ks.shape[1]
    params = [getattr(reader,k).to(device) for k in ('A','B','bk','bv')]
    error = signal = 0.0
    kls,masses = [],[]
    with torch.no_grad():
        for tr,sr in rows:
            s,t = [_group(r,target,target_head,heads,device) for r in (sr,tr)]
            ids = sr['layers'][target]['selected_tokens'].to(device)
            so,sp,sb = old.distribution(s['q'],ks[ids,source_head:source_head+1],
                vs[ids,source_head:source_head+1],s['tail_output'],s['tail_lse'],*params)
            _,tp,tb = old.distribution(t['q'],kt[ids,target_head:target_head+1],
                vt[ids,target_head:target_head+1],t['tail_output'],t['tail_lse'])
            error += (so-t['attention_output']).double().square().sum().item()
            signal += t['attention_output'].double().square().sum().item()
            kls.append(max(0.0,float((tp.exp()*(tp-sp)).sum(-1).mean())))
            masses.append((sb-tb).abs().flatten().cpu())
    rmse = (error/max(signal,1e-30))**0.5
    return {'output_relative_rmse':rmse,'joint_attention_kl':sum(kls)/len(kls),
            'prefix_mass_p95':float(torch.quantile(torch.cat(masses),0.95)),
            'requests':len(rows)}


def fit_assignment(rows,target,ks,vs,kt,vt,*,ridge,device,key_tokens=8192,
                   key_mode='dense',rank=16,seed=22,kl_weight=1.0,incumbent=None):
    edge_fit,edge_select = split_fit(rows,seed=seed)
    heads = ks.shape[1]
    cost = torch.zeros(heads,heads,dtype=torch.float64)
    reports,readers = {},{}
    for h in range(heads):
        for s in range(heads):
            r = fit_edge(edge_fit,target,s,h,ks,vs,kt,vt,ridge=ridge,device=device,
                key_tokens=key_tokens,key_mode=key_mode,rank=rank,
                incumbent=incumbent if h==s else None)
            report = edge_report(edge_select,target,s,h,ks,vs,kt,vt,r,device)
            cost[h,s] = report['output_relative_rmse']**2+kl_weight*report['joint_attention_kl']
            readers[h,s],reports[f'{h}:{s}'] = r,report
    result = choose_assignment(cost)
    result.update(cost_matrix=cost.tolist(),edge_reports=reports,
                  edge_fit_query_hashes=[r[0]['query_hash'] for r in edge_fit],
                  assignment_query_hashes=[r[0]['query_hash'] for r in edge_select],
                  cost_definition='per-group output relative MSE + kl_weight * joint attention KL',
                  kl_weight=kl_weight)
    return result,readers


def _canonical_readers(bundle):
    return {int(l):Reader.from_state_dict(r).to(device='cpu',dtype=torch.bfloat16)
            for l,r in bundle.get('readers',{}).items()}


def _validate_data(tf,tv,sf,sv,target,source,carry):
    assert_splits(tf,tv); assert_splits(sf,sv)
    fit,val = paired(tf,sf,target),paired(tv,sv,target)
    header = next(iter(tf.values()));g=header['geometry']
    if not 0 <= target < g['layers'] or not 0 <= source < g['layers']:
        raise ValueError('Out-of-range layer')
    for dataset in (tf,tv,sf,sv):
        for r in dataset.values():
            for f in IDENTITY_FIELDS:
                if r[f]!=header[f]: raise ValueError(f'Mixed identity: {f}')
    for dataset in (tf,tv):
        if next(iter(dataset.values()))['rollout_sources']!=list(range(g['layers'])):
            raise ValueError('Teacher must be clean identity rollout')
    sources = list(range(g['layers'])) if carry is None else list(carry['sources'])
    readers = {} if carry is None else _canonical_readers(carry)
    maps = normalized_maps({} if carry is None else carry.get('head_maps',{}),readers,g['kv_heads'])
    expected = aligned_fingerprint(readers,maps)
    for dataset in (sf,sv):
        row = next(iter(dataset.values()))
        if row['rollout_sources']!=sources or row['rollout_reader_fingerprint']!=expected:
            raise ValueError('Student trace differs from --carry, including upstream head maps')
    if carry is not None:
        for f in IDENTITY_FIELDS:
            if carry[f]!=header[f]: raise ValueError(f'Carry identity mismatch: {f}')
    if any(l>=target for l in readers):
        raise ValueError('Add consumers in increasing target-layer order; re-record downstream states')
    if target==source or sources[target]!=target or sources[source]!=source:
        raise ValueError('Target must be new and source must own its physical KV')
    if target in sources[:target]: raise ValueError('Target already serves an earlier consumer')
    proposed=sources.copy();proposed[target]=source
    validate_sources(proposed,period=header['period'])
    for _,r in fit+val:
        if r['selection_by_layer'][source]!=r['selection_by_layer'][target]:
            raise ValueError('Source/target selected IDs differ; no hidden union')
    return fit,val,header,proposed,readers,maps


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('store-root','task','teacher-fit','teacher-val','student-fit','student-val','output'):
        p.add_argument('--'+x,required=True)
    p.add_argument('--source',type=int,required=True);p.add_argument('--target',type=int,required=True)
    p.add_argument('--mapping',choices=('search','identity'),default='search')
    p.add_argument('--carry');p.add_argument('--initial-reader')
    p.add_argument('--device',default='cpu');p.add_argument('--steps',type=int,default=0)
    p.add_argument('--eval-every',type=int,default=20);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--ridge',type=float,default=1e-3);p.add_argument('--tokens',type=int,default=8192)
    p.add_argument('--seed',type=int,default=22);p.add_argument('--key-mode',choices=('dense','rope_rr'),default='dense')
    p.add_argument('--rank',type=int,default=16);p.add_argument('--kl-weight',type=float,default=1.)
    p.add_argument('--post-weight',type=float,default=1.);p.add_argument('--trust-weight',type=float,default=1e-3)
    p.add_argument('--bf16-ste',action='store_true')
    a=p.parse_args(argv)
    if Path(a.output).exists():raise FileExistsError(a.output)
    if a.steps<0 or min(a.ridge,a.lr,a.tokens,a.eval_every)<=0 or min(a.kl_weight,a.post_weight,a.trust_weight)<0:
        raise ValueError('Invalid calibration parameters')
    torch.set_num_threads(min(4,torch.get_num_threads()));torch.manual_seed(a.seed);rng=random.Random(a.seed)
    tf,tv,sf,sv=[load_traces(d,a.task) for d in (a.teacher_fit,a.teacher_val,a.student_fit,a.student_val)]
    carry=load_safe(a.carry) if a.carry else None
    fit,val,header,sources,carried,maps=_validate_data(tf,tv,sf,sv,a.target,a.source,carry)
    arrays=[]
    for l,k in ((a.source,'key'),(a.source,'value'),(a.target,'key'),(a.target,'value')):
        x,m=read_prefix(a.store_root,a.task,l,k)
        if m['token_hash']!=header['token_hash']:raise ValueError('Wrong prefix')
        arrays.append(x)
    device=torch.device(a.device);ks,vs,kt,vt=[x.to(device) for x in arrays]
    incumbent=None; init_bundle=None
    if a.initial_reader:
        init_bundle=load_safe(a.initial_reader)
        for f in IDENTITY_FIELDS:
            if init_bundle[f]!=header[f]:raise ValueError(f'Initial reader mismatch: {f}')
        if init_bundle['sources'][a.target]!=a.source:raise ValueError('Initial source mismatch')
        im=init_bundle.get('head_maps',{}).get(str(a.target),list(range(ks.shape[1])))
        if im!=list(range(ks.shape[1])):raise ValueError('Incumbent must be a same-index reader')
        incumbent=_canonical_readers(init_bundle)[a.target].to(device='cpu',dtype=torch.float32)
    # Both search and identity controls measure the same assignment grid and use the same fit budget.
    assignment,edges=fit_assignment(fit,a.target,ks,vs,kt,vt,ridge=a.ridge,device=device,
        key_tokens=a.tokens,key_mode=a.key_mode,rank=a.rank,seed=a.seed,
        kl_weight=a.kl_weight,incumbent=incumbent)
    heads=ks.shape[1]
    permutation=assignment['permutation'] if a.mapping=='search' else list(range(heads))
    reader=Reader(*(torch.cat([getattr(edges[h,permutation[h]],k) for h in range(heads)],0)
                     for k in ('A','B','bk','bv')))
    pi=torch.tensor(permutation,device=device)
    # OFFLINE ONLY: reorder source arrays for reusing v2.1 objectives. Runtime never does this.
    mkp,mvp=ks.index_select(1,pi),vs.index_select(1,pi)
    reader=old.fit_value_asymmetric(fit,a.target,mkp,mvp,kt,vt,reader,ridge=a.ridge,device=device)
    weights=load_safe(Path(a.teacher_fit)/'weights'/f'layer_{a.target:02d}.pt')
    if weights['model_identity']!=header['model_identity']:raise ValueError('o_proj identity mismatch')
    wo=weights['o_proj'].float().to(device).detach()
    module=old.TrainableReader(reader,ste=a.bf16_ste).to(device)
    initial=[x.detach().clone() for x in module.parameters()]
    best=module.export()
    def evaluate(r):return old.evaluate(val,a.target,mkp,mvp,kt,vt,r,device,wo)
    def score(r):return r['output_relative_rmse']**2+a.kl_weight*r['mean_joint_attention_kl']+a.post_weight*r['mean_post_projection_update_relative_rmse']**2
    report=evaluate(best);best_score=score(report);history=[{'step':0,**report}]
    opt=torch.optim.Adam(module.parameters(),lr=a.lr)
    for step in range(1,a.steps+1):
        row=fit[rng.randrange(len(fit))]
        so,sp,_,to,tp,_,t,s=old.sample_outputs(row,a.target,mkp,mvp,kt,vt,module.tensors(),device)
        att,kl,post=old.objective(so,sp,to,tp,student_hidden=s['hidden_in'],teacher_pre_mlp=t['pre_mlp'],
                                 teacher_hidden=t['hidden_in'],o_weight=wo)
        trust=sum((x-x0).square().sum() for x,x0 in zip(module.parameters(),initial))/sum(x0.square().sum() for x0 in initial).clamp_min(1.)
        loss=att+a.kl_weight*kl+a.post_weight*post+a.trust_weight*trust
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(module.parameters(),1.);opt.step()
        if step%a.eval_every==0 or step==a.steps:
            current=module.export();r=evaluate(current);history.append({'step':step,**r})
            if score(r)<best_score:best,best_score=current,score(r)
    report=evaluate(best)
    local_pass=(report['output_relative_rmse']<=.03 and report['max_request_output_relative_rmse']<=.08
                and report['prefix_mass_p95']<=.03)
    out_readers={str(l):r.state_dict() for l,r in carried.items()};out_readers[str(a.target)]=best.state_dict()
    maps[a.target]=tuple(permutation)
    banned=set(tf)|set(tv)
    for b in (carry,init_bundle):
        if b is not None:banned.update(b['calibration_query_hashes'])
    bundle={'format':FORMAT,'base_commit':BASE_COMMIT,'audited_base_v2':BASE_V2,
        **{f:header[f] for f in IDENTITY_FIELDS},'sources':sources,'readers':out_readers,
        'head_maps':{str(l):list(v) for l,v in maps.items()},
        'calibration_query_hashes':sorted(banned),'diagnostic_only':True,
        'deployment_status':'diagnostic_candidate_not_accepted','end_to_end_validated':False,
        'local_gate_passed':local_pass,'last_consumer':a.target,'report':report,
        'assignment':assignment,'compiler_args':vars(a),
        'warning':'Assignment-set improvement is not independent quality or speed certification; old v21 runtime must reject this format.'}
    save_new(a.output,bundle)
    Path(a.output).with_suffix('.report.json').write_text(json.dumps({
        'head_map':permutation,'assignment':assignment,'final':report,'history':history,
        'local_gate_passed':local_pass,'args':vars(a)},indent=2,allow_nan=False))
    print(json.dumps({'output':a.output,'head_map':permutation,'local_gate_passed':local_pass,**report},indent=2))

if __name__=='__main__':main()
