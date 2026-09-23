#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contiguous_fuxian.sharedkv_v26.analysis import records,paired_time_blocks
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
s=json.loads(Path(a.manifest).read_text());index={}
for j in s['jobs']:
 index[(j['condition'],j['block'],j['arm'])]=records(Path(j['output'])/'scored_records.jsonl')
results={}
for condition in s['spec'].get('conditions',['preserve','warm']):
 for admission in ('original','reuse'):
  pairs=[{'block':b,'baseline':index[(condition,b,'I-'+admission)],'student':index[(condition,b,'S-'+admission)]}
         for b in range(s['spec']['blocks'])]
  results[condition+'/'+admission]=paired_time_blocks(pairs)
 # Admission optimization must preserve classification in BOTH model variants.
 for mode in ('I','S'):
  mismatches=[]
  for b in range(s['spec']['blocks']):
   x,y=index[(condition,b,mode+'-original')],index[(condition,b,mode+'-reuse')]
   if set(x)!=set(y):raise ValueError('Admission parity queries differ')
   for key in x:
    if 'prediction' not in x[key] or 'prediction' not in y[key]:raise ValueError('Missing prediction evidence')
    if x[key].get('prediction')!=y[key].get('prediction') or x[key]['correct']!=y[key]['correct']:
     mismatches.append({'block':b,'uid':key})
  results[condition+'/'+mode+'/admission_prediction_parity']={'mismatches':mismatches,
      'note':'Prediction equality is weaker than full logits/selected-IDs/cache parity; run those audits separately.'}
with Path(a.output).open('x') as f:json.dump(results,f,indent=2,allow_nan=False)
