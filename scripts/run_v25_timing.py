
import json
from v25_common import *
from v25_environment import snapshot
chosen=json.loads((ROOT/'audit/v25_frozen_selection.json').read_text())['chosen_G']
arms=[('identity',None,None)]+[(f'g{G}_r{R}',G,R) for G in [26,24,22,20,16,14] for R in [0,4]]
selected=[('identity',None,None),(f'g{chosen}_r0',chosen,0),(f'g{chosen}_r4',chosen,4)]
orders=[[0,1,2],[2,1,0],[1,0,2],[2,0,1]]
records=[]
def one(policy,block,arm,n):
 label,G,R=arm;name=f'v25_time_{policy}_b{block}_{label}'
 a=args(name,G=G,R=R,n=n,warm=8,gpu=3,cold=None if policy=='warm' else policy)
 records.append({'name':name,'policy':policy,'block':block,'arm':label,'G':G,'R':R,'n':n,'args':a})
 snapshot(name,'before')
 p=suite(name,[a],3);wait_all([p]);snapshot(name,'after');print('TIMING DONE',name,flush=True)
# Fixed finite exploration curve, two whole-grid forward/reverse blocks.
for b,ordered in [(0,arms),(1,list(reversed(arms)))]:
 for arm in ordered:one('preserve',b,arm,16)
# Confirm both R routes at the selected G, without adding/removing old blocks.
for b in [2,3]:
 for i in orders[b]:one('preserve',b,selected[i],16)
for policy,n in [('native',16),('warm',32)]:
 for b,order in enumerate(orders):
  for i in order:one(policy,b,selected[i],n)
with (ROOT/'audit/v25_formal_timing_manifest.json').open('x') as f:json.dump(records,f,indent=2)
print('ALL V25 FORMAL TIMING COMPLETE',flush=True)
