#!/usr/bin/env python3
"""Recompute resource/latency distinctions from published v1 comparison.csv."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--csv',type=Path,default=Path('docs/layerkv/results/comparison.csv'))
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
rows={r['variant']:r for r in csv.DictReader(a.csv.open())}
required=['original','identity','dense1','dense3','dense7','identity_fixed_v2','dense1_fixed_v2']
for n in required:
    if n not in rows:raise ValueError(f'Missing published row: {n}')
def compare(b,s):
    x,y=rows[b],rows[s]
    if int(x['n'])!=int(y['n']):raise ValueError('Do not compare unmatched sample counts')
    result={'baseline':b,'student':s,'n':int(x['n'])}
    for field in ['mean_ttft_ms','mean_physical_prefetch_bytes','mean_host_source_payload_bytes','mean_selector_bytes']:
        u,v=float(x[field]),float(y[field]);result[field]={'baseline':u,'student':v,'delta':v-u,'reduction_pct':100*(u-v)/u if u else None}
    u=float(x['mean_host_source_payload_bytes'])+float(x['mean_selector_bytes'])
    v=float(y['mean_host_source_payload_bytes'])+float(y['mean_selector_bytes'])
    result['host_payload_plus_selector_proxy_reduction_pct']=100*(u-v)/u
    result['warning']='Source-counter proxy, NOT hardware H2D. Selector addition assumes its source is host; verify native metrics.'
    return result
r={'source':'utazh/pans v1 @59312f2bd2504079fed03b53f283d8ade931fee2',
   'published_comparisons':[compare('identity','dense1'),compare('identity_fixed_v2','dense1_fixed_v2'),compare('original','identity')],
   'whole_plan_exact_matches_identity_vs_original':round(float(rows['identity']['selection_agreement_vs_original'])*int(rows['identity']['n'])),
   'note':'Whole-plan exact hash fraction is not token/head/layer overlap. Fixed 32 and online128 are separate experiments.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
