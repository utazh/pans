"""Publication-style static plots for actual v24 measurements."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path(__file__).resolve().parents[1];d=json.loads((r/'results/v24_final_analysis.json').read_text())
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors=['#497ca6','#d58b2b','#43866b']
dev=d['quality']['dev'];base=dev['identity']['basic']['accuracy_pct']
for x,(prefix,label) in enumerate([('group_s','Group output'),('score_g26_s','Final score + margin'),('nomargin_g26_s','Final score only')]):
 vals=[dev[prefix+str(s)]['basic']['accuracy_pct'] for s in [24,25,26]]
 ax[0,0].scatter([x-.06,x,x+.06],vals,color=colors[x],s=55,zorder=3)
 ax[0,0].plot([x-.16,x+.16],[np.mean(vals)]*2,color=colors[x],lw=2)
ax[0,0].axhline(base,color='black',ls='--',lw=1,label=f'Identity {base:.2f}%')
ax[0,0].set_xticks(range(3),['Group output','Score + margin','Score only'])
ax[0,0].set_ylabel('Accuracy (%)');ax[0,0].set_title('Fresh development: 768 queries, three seeds');ax[0,0].legend(fontsize=8)
G=d['frozen_selection']['chosen_G']
for x,split in enumerate(['final768','test500']):
 q=d['quality'][split];b=q['identity']['basic']['accuracy_pct']
 vals=[q[f'score_g{G}_s{s}']['basic']['accuracy_pct'] for s in [24,25,26]]
 ax[0,1].scatter([x-.08,x,x+.08],vals,color=colors[1],s=55,zorder=3)
 ax[0,1].scatter([x],[b],color='black',marker='_',s=400,label='Identity' if x==0 else None)
 ax[0,1].plot([x-.17,x+.17],[np.mean(vals)]*2,color=colors[1],lw=2)
ax[0,1].set_xticks([0,1],['Fresh final 768','Historical TREC 500'])
ax[0,1].set_ylabel('Accuracy (%)');ax[0,1].set_title(f'Frozen G{G} final-score readers');ax[0,1].legend(fontsize=8)
for x,policy in enumerate(['native','preserve']):
 t=d['timing'][policy]['all']
 for off,key,color in [(-.18,'identity_ms',colors[0]),(.18,'shared_ms',colors[1])]:
  vals=[b[key] for b in t['per_block']]
  ax[1,0].bar(x+off,np.mean(vals),width=.32,color=color,alpha=.65,label=('Identity' if off<0 else f'Shared G{G}') if x==0 else None)
  ax[1,0].scatter(np.full(len(vals),x+off),vals,color=color,edgecolor='white',s=30,zorder=3)
ax[1,0].set_xticks([0,1],['SSD-cold / native','SSD-cold / preserve'])
ax[1,0].set_ylabel('Mean TTFT (ms)');ax[1,0].set_title('Four paired blocks per policy, 32 queries each');ax[1,0].legend(fontsize=8)
for i,policy in enumerate(['warm','native','preserve']):
 vals=[100*(b['shared_ms']/b['identity_ms']-1) for b in d['timing'][policy]['all']['per_block']]
 ax[1,1].plot(range(1,5),vals,marker='o',color=colors[i],label=policy)
vals=[100*(b['shared_ms']/b['identity_ms']-1) for b in d['warm_fresh_process_confirmation']['all']['per_block']]
ax[1,1].plot(range(1,5),vals,marker='s',ls='--',color='#666666',label='fresh warm (post hoc)')
ax[1,1].axhline(0,color='black',ls='--',lw=1)
ax[1,1].set_xticks(range(1,5),['1: I-S','2: S-I','3: S-I','4: I-S'])
ax[1,1].set_ylabel('Shared TTFT change (%)');ax[1,1].set_title('Order sensitivity; negative is faster');ax[1,1].legend(fontsize=8)
for a in ax.flat:a.grid(axis='y',alpha=.2);a.spines[['top','right']].set_visible(False)
fig.suptitle('Qwen2.5-7B / TREC / 25% budget / fixed 4,940-token prefix',fontsize=13)
(r/'figures').mkdir(exist_ok=True)
fig.savefig(r/'figures/v24_results.png',dpi=180)
fig.savefig(r/'figures/v24_results.pdf')
print('Saved actual-result figure')
