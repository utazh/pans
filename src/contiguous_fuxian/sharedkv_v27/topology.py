"""Bounded-distance alternatives, NOT a new sharing/quality theorem.

Given clean FIT selection plans, enumerate small P8 windows and choose owned
source maps satisfying exact IDs, native leaders, protected early layers and
a fanout cap. Never widen the distance cap to force G16. Return infeasible Gs.
"""
from __future__ import annotations
import itertools
from collections import Counter
from .common import validate_sources

def compatible(rows,n):
 if not rows:raise ValueError('Need fit data')
 table=[[True]*n for _ in range(n)]
 for row in rows:
  if row.get('split')!='fit':raise ValueError('Topology design is FIT-only')
  p=row['selection_by_layer']
  if len(p)!=n:raise ValueError('Wrong plan length')
  for i in range(n):
   for j in range(n):table[i][j]&=p[i]==p[j]
 return table

def _assign(consumers,owners,allowed,fanout):
 # At most 8 nodes. Dynamic recursion keeps the exact minimal distance cost.
 consumers=sorted(consumers,key=lambda i:(sum(allowed[i][s] for s in owners),i))
 best=None;counts=Counter();mapping={}
 def visit(at,cost):
  nonlocal best
  if best is not None and cost>best[0]:return
  if at==len(consumers):
   ordered=tuple(sorted(mapping.items()));item=(cost,ordered)
   if best is None or item<best:best=item
   return
  i=consumers[at]
  for s in sorted(owners,key=lambda s:(abs(i-s),s)):
   if allowed[i][s] and counts[s]<fanout:
    counts[s]+=1;mapping[i]=s;visit(at+1,cost+abs(i-s));counts[s]-=1;del mapping[i]
 visit(0,0)
 return best

def bounded_plans(rows,n,period,*,max_distance=1,max_consumers=2,protect_first=4):
 if n<1 or not 1<=period<=8 or type(max_distance) is not int or max_distance<1 or max_consumers<1 or not 0<=protect_first<=n:
  raise ValueError('Invalid explicit topology limits; period enumeration capped at 8')
 ok=compatible(rows,n)
 allow=[[ok[i][j] and abs(i-j)<=max_distance for j in range(n)] for i in range(n)]
 dp={0:(0,[])}
 for start in range(0,n,period):
  nodes=list(range(start,min(n,start+period)));local={}
  mandatory={i for i in nodes if i<protect_first or i%period==0}
  optional=[i for i in nodes if i not in mandatory]
  for bits in itertools.product((False,True),repeat=len(optional)):
   owners=sorted(mandatory|{i for i,on in zip(optional,bits) if on})
   consumers=[i for i in nodes if i not in owners]
   solved=_assign(consumers,owners,allow,max_consumers)
   if solved is None:continue
   cost,pairs=solved;mp={i:i for i in nodes};mp.update(dict(pairs))
   mapping=tuple(mp[i] for i in nodes)
   k=len(owners);item=(cost,mapping)
   if k not in local or item<local[k]:local[k]=item
  out={}
  for g,(cost,prev) in dp.items():
   for k,(lc,mp) in local.items():
    item=(cost+lc,prev+list(mp))
    if g+k not in out or item<out[g+k]:out[g+k]=item
  dp=out
 for g,(cost,mp) in dp.items():validate_sources(mp,period)
 return {g:{'sources':mp,'distance_sum':cost,'max_distance_actual':max(abs(i-s) for i,s in enumerate(mp)),
    'complete_sources':g,'consumers':n-g} for g,(cost,mp) in sorted(dp.items())}

def describe(plan,sources):
 validate_sources(sources)
 n=len(sources)
 if len(plan)!=n:raise ValueError('Bad length')
 return {'sources':len(set(sources)),'consumers':sum(i!=s for i,s in enumerate(sources)),
   'max_distance':max(abs(i-s) for i,s in enumerate(sources)),
   'mean_consumer_distance':sum(abs(i-s) for i,s in enumerate(sources))/max(1,sum(i!=s for i,s in enumerate(sources))),
   'fanout':dict(Counter(s for i,s in enumerate(sources) if i!=s)),
   'equal_support':all(plan[i]==plan[s] for i,s in enumerate(sources))}
