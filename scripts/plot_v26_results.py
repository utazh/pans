
from pathlib import Path
import sys,json
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'runtime_deps_plot'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
rows=json.loads((r/'results/v26_matrix_analysis.json').read_text())['matrix']
fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
tasks=['sst2','subj','trec','rte'];budgets=[5,10,25,50]
for ax,key,title in zip(axes,['delta_pp','preserve_delta_pct','warm_delta_pct'],['Accuracy change (percentage points)','SSD-cold TTFT change (%)','Ordinary-cache TTFT change (%)']):
 arr=np.array([[next(x[key] for x in rows if x['prefix']=='p0' and x['task']==t and x['budget']==b) for b in budgets] for t in tasks])
 limit=max(1,float(np.abs(arr).max()))
 ax.imshow(arr,cmap='RdBu' if key=='delta_pp' else 'RdBu_r',vmin=-limit,vmax=limit)
 for i in range(4):
  for j in range(4):ax.text(j,i,f'{arr[i,j]:+.2f}',ha='center',va='center',color='white' if abs(arr[i,j])>limit*.65 else '#111827',fontsize=13)
 ax.set_xticks(range(4),[f'{b}%' for b in budgets]);ax.set_yticks(range(4),[x.upper() for x in tasks]);ax.set_title(title,fontsize=12);ax.set_xlabel('KV budget')
fig.suptitle('Qwen2.5-7B: G16 / R4 vs native KV | same reuse-prefetch baseline',fontsize=15)
fig.text(.5,-.025,'Blue indicates improvement. Full evaluation for accuracy; 128 paired queries per timing cell. Matrix timing is descriptive, not an order-independent speed certificate.',ha='center',fontsize=10)
out=r/'results/v26_matrix.png';assert not out.exists();fig.savefig(out,dpi=180,bbox_inches='tight');plt.close(fig)
