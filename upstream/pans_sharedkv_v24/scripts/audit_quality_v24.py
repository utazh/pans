#!/usr/bin/env python3
"""Paired development report; never auto-promotes a reader or a source plan."""
import argparse,json
from pathlib import Path
from scipy.stats import beta

def records(path):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    out={}
    for r in rows:
        key=(r.get('task',''),str(r['uid']))
        if key in out:raise ValueError('Duplicate task/UID; do not pool repeat timing runs as quality samples')
        if type(r['correct']) is not bool:raise ValueError('correct must be a Boolean')
        out[key]=r
    if not out:raise ValueError('No quality records')
    return out

def paired_report(a,b,margin=.01,alpha=.05):
    if set(a)!=set(b):raise ValueError('Quality samples must be paired exactly')
    n=len(a);harm=sum(a[k]['correct'] and not b[k]['correct'] for k in a)
    benefit=sum(not a[k]['correct'] and b[k]['correct'] for k in a)
    # Bonferroni lower bound on p(benefit)-p(harm). Conservative, but avoids
    # claiming zero risk from a degenerate bootstrap with no observed flips.
    lower_b=0. if benefit==0 else float(beta.ppf(alpha/2,benefit,n-benefit+1))
    upper_h=1. if harm==n else float(beta.ppf(1-alpha/2,harm+1,n-harm))
    return {'n':n,'baseline_correct':sum(r['correct'] for r in a.values()),
        'student_correct':sum(r['correct'] for r in b.values()),'harmful':harm,'beneficial':benefit,
        'net_delta_pp':100*(benefit-harm)/n,'conservative_one_sided_lower_pp':100*(lower_b-upper_h),
        'noninferiority_margin_pp':100*margin,'conservative_bound_passes':lower_b-upper_h>=-margin,
        'assumption':'Independent query sample, fixed predeclared comparison; not corrected for topology/model selection',
        'deployment_status':'diagnostic_report_only_no_auto_promotion'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True);p.add_argument('--student',required=True)
    p.add_argument('--margin',type=float,default=.01);p.add_argument('--output',required=True)
    a=p.parse_args()
    if not 0<a.margin<1:raise ValueError('Margin must be predeclared as a fraction')
    out=Path(a.output)
    with out.open('x') as f:json.dump(paired_report(records(a.baseline),records(a.student),a.margin),f,indent=2)
if __name__=='__main__':main()
