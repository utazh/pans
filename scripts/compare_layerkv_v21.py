#!/usr/bin/env python3
"""Pair actual v1 scored_records and optional logits. No silent intersection.

TTFT differences are descriptive for this run, NOT independent repeated trials.
Whole-plan hash agreement is NOT token Jaccard. Logit capture may perturb
cross-request execution; final latency runs should be uncaptured.
"""
import argparse,json,sys,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contiguous_fuxian.sharedkv_v21.audit import paired_quality,mean_label_scores

def index(root):
    out={}
    for line in (Path(root)/'scored_records.jsonl').read_text().splitlines():
        r=json.loads(line);key=(str(r.get('task','')),str(r['uid']))
        if key in out:raise ValueError('Duplicate scored UID')
        out[key]=r
    if not out:raise ValueError('No scored requests')
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True);p.add_argument('--student',required=True)
    p.add_argument('--baseline-logits');p.add_argument('--student-logits')
    p.add_argument('--margin-pp',type=float,default=1.)
    p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    b,s=index(a.baseline),index(a.student)
    if set(b)!=set(s):raise ValueError('Different scored query sets; subset explicitly before comparing')
    def qr(x):return [dict(task=k[0],query_hash=k[1],correct=int(r['correct'])) for k,r in x.items()]
    result={'quality':paired_quality(qr(b),qr(s),margin_pp=a.margin_pp),
        'scope':'One paired run. No run-to-run speed confidence claim.'}
    for name,x in [('baseline',b),('student',s)]:
        result[name]={'mean_ttft_ms':statistics.mean(r['ttft_ms'] for r in x.values()),'n':len(x)}
        for field in ['physical_prefetch_kv_bytes','selected_kv_bytes','selector_key_bytes',
                      'prefetch_cpu_source_tensor_tokens','prefetch_disk_source_tensor_tokens']:
            if all(field in r for r in x.values()):result[name][field]=statistics.mean(r[field] for r in x.values())
    result['mean_paired_ttft_delta_ms']=statistics.mean(s[k]['ttft_ms']-b[k]['ttft_ms'] for k in b)
    result['whole_plan_exact_fraction']=statistics.mean(b[k].get('layer_token_selection_sha256')==s[k].get('layer_token_selection_sha256') for k in b) if all('layer_token_selection_sha256' in r for r in list(b.values())+list(s.values())) else None
    if bool(a.baseline_logits)!=bool(a.student_logits):raise ValueError('Supply both logits directories')
    if a.baseline_logits:
        import torch
        from contiguous_fuxian.sharedkv.io import load_safe
        bp={p.stem:p for p in Path(a.baseline_logits).glob('*.pt')};sp={p.stem:p for p in Path(a.student_logits).glob('*.pt')}
        if set(bp)!=set(sp) or not bp:raise ValueError('Different/empty captured query sets')
        # Captures can include warmup; do not silently call their n the scored n.
        kl=[];jac={};labels=[]
        for h in sorted(bp):
            x,y=load_safe(bp[h]),load_safe(sp[h]);lx=x['logits'].float().log_softmax(-1);ly=y['logits'].float().log_softmax(-1)
            kl.append(float((lx.exp()*(lx-ly)).sum()))
            if len(x['selected_by_layer'])!=len(y['selected_by_layer']):raise ValueError('Different layer count')
            for lid,(ia,ib) in enumerate(zip(x['selected_by_layer'],y['selected_by_layer'])):
                u,v=set(ia),set(ib);jac.setdefault(str(lid),[]).append(len(u&v)/max(1,len(u|v)))
            ax,ay=x.get('label_token_logprobs'),y.get('label_token_logprobs')
            if ax is not None and ay is not None:
                if set(ax)!=set(ay):raise ValueError('Different label continuation sets')
                if len(ax)<2:raise ValueError('At least two labels required for a margin')
                keys=sorted(ax);xx=torch.tensor([mean_label_scores(ax)[k] for k in keys]);yy=torch.tensor([mean_label_scores(ay)[k] for k in keys])
                vals=xx.sort(descending=True).values
                labels.append({'query_hash':h,'teacher_label_margin':float(vals[0]-vals[1]),
                    'label_score_max_abs_delta':float((xx-yy).abs().max()),'label_argmax_agreement':bool(xx.argmax()==yy.argmax()),
                    'normalized_label_score_kl':float((xx.softmax(-1)*(xx.log_softmax(-1)-yy.log_softmax(-1))).sum())})
        result['captures']={'n':len(bp),'warning':'Capture count may include warmup; scored accuracy n is separate',
            'first_token_full_vocab_mean_kl':statistics.mean(kl),
            'token_jaccard_by_layer':{k:{'mean':statistics.mean(v),'min':min(v)} for k,v in jac.items()},
            'label_score_reduction':'mean per token, matching original runner','label_continuation_diagnostics':labels}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
