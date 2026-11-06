"""Paired effects and honest Pareto summaries; NOT automatic certification."""
from __future__ import annotations
import json,math
from pathlib import Path
from scipy.stats import beta

def load_records(path):
 records=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
 out={}
 for r in records:
  uid=str(r['uid'])
  if uid in out:raise ValueError('Repeated UID is not an independent quality sample')
  if not isinstance(r['correct'],(bool,int)) or r['correct'] not in (True,False,0,1):raise ValueError('Invalid correct field')
  out[uid]=r
 if not out:raise ValueError('Empty records')
 return out

def paired_quality(base,student,alpha=.05):
 if set(base)!=set(student) or not base:raise ValueError('Need identical nonempty UID sets')
 if not 0<alpha<1:raise ValueError('Bad alpha')
 n=len(base);bad=good=0
 for uid,b in base.items():
  s=student[uid]
  for k in ('task','gold','label','answer','query_hash'):
   if k in b and k in s and b[k]!=s[k]:raise ValueError(f'Paired identity mismatch: {k}')
  bad+=int(bool(b['correct']) and not bool(s['correct']))
  good+=int(not bool(b['correct']) and bool(s['correct']))
 lower_good=0. if good==0 else float(beta.ppf(alpha/2,good,n-good+1))
 upper_bad=1. if bad==n else float(beta.ppf(1-alpha/2,bad+1,n-bad))
 return {'n':n,'baseline_accuracy':sum(bool(r['correct']) for r in base.values())/n,
   'student_accuracy':sum(bool(r['correct']) for r in student.values())/n,
   'harmful':bad,'beneficial':good,'delta_pp':100*(good-bad)/n,
   'conservative_lower_pp':100*(lower_good-upper_bad),'alpha':alpha,
   'certified':False,'note':'Marginal binomial bounds with within-pair union bound. No post-selection/multi-seed correction implied. Preserve old preregistered verdicts.'}

def pareto(candidates):
 # Returns observed, same-protocol non-dominated points. Never fit/test interpolation.
 if not candidates:return []
 protocol_keys=('task','prefix_id','model_id','eval_uid_digest','cache_condition','admission')
 first=candidates[0]
 for c in candidates:
  for k in protocol_keys:
   if k not in c or c[k]!=first[k]:raise ValueError(f'Mixed or missing Pareto protocol: {k}')
  for k in ('accuracy','ttft_ms','subsystem_bytes'):
   if not math.isfinite(c[k]):raise ValueError('Nonfinite result')
  if not 0<=c['accuracy']<=1 or min(c['ttft_ms'],c['subsystem_bytes'])<=0:raise ValueError('Invalid units')
 def dominates(a,b):
  vals=(a['accuracy']>=b['accuracy'],a['ttft_ms']<=b['ttft_ms'],a['subsystem_bytes']<=b['subsystem_bytes'])
  return all(vals) and (a['accuracy']>b['accuracy'] or a['ttft_ms']<b['ttft_ms'] or a['subsystem_bytes']<b['subsystem_bytes'])
 return [c for c in candidates if not any(dominates(a,c) for a in candidates)]
