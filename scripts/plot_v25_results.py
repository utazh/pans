
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];d=json.loads((r/'results/v25_final_analysis.json').read_text());G=d['frozen_selection']['chosen_G'];gs=[26,24,22,20,16,14];colors={0:'#2563eb',4:'#ea580c'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
for ax,split,title in [(axes[0,0],'dev','Reused development: 768 questions'),(axes[0,1],'test500','Historical TREC test: 500 questions')]:
 for R in [0,4]:ax.plot(gs,[d['quality'][split][f'g{g}_r{R}_s24']['basic']['accuracy_pct'] for g in gs],'-o',color=colors[R],label=f'R={R}')
 ax.axhline(d['quality'][split]['identity']['basic']['accuracy_pct'],ls='--',color='black',label='Identity')
 ax.set(xlabel='Full prefix KV sources (G)',ylabel='Accuracy (%)',title=title,xticks=gs,xlim=(27,13));ax.legend()
ax=axes[0,2]
for R in [0,4]:
 ax.plot(gs,[100*d['resource'][f'g{g}_r{R}']['selected_payload_reduction_fraction'] for g in gs],'-o',color=colors[R],label=f'R={R}: payload')
 ax.plot(gs,[d['resource'][f'g{g}_r{R}']['selected_plus_readers_net_reduction_pct'] for g in gs],'--',color=colors[R],label=f'R={R}: incl. readers')
ax.set(xlabel='Full prefix KV sources (G)',ylabel='Reduction (%)',title='Actual selected buffers; model/disk excluded',xticks=gs,xlim=(27,13));ax.legend(fontsize=8)
ax=axes[1,0]
for R in [0,4]:ax.plot(gs,[d['curve_preserve'][f'g{g}_r{R}']['shared_mean_ms'] for g in gs],'-o',color=colors[R],label=f'R={R}')
ax.axhline(d['curve_preserve']['g26_r0']['baseline_mean_ms'],ls='--',color='black',label='Identity')
ax.set(xlabel='Full prefix KV sources (G)',ylabel='TTFT (ms)',title='SSD-cold/preserve: exploratory 2 × 16',xticks=gs,xlim=(27,13));ax.legend()
ax=axes[1,1]
for R in [0,4]:
 vals=[d['quality']['final'][f'g{G}_r{R}_s{s}']['paired_vs_identity'] for s in [24,25,26]]
 x=np.arange(3)+(R/4-.5)*.18
 ax.scatter(x,[v['net_delta_pp'] for v in vals],color=colors[R],label=f'R={R}',s=40)
 for xx,v in zip(x,vals):ax.plot([xx,xx],[v['conservative_one_sided_lower_pp'],v['net_delta_pp']],color=colors[R])
ax.axhline(0,color='black',lw=.7);ax.axhline(-1,ls='--',color='red',label='NI margin: −1 pp')
ax.set(xticks=[0,1,2],xticklabels=['Seed 24','Seed 25','Seed 26'],ylabel='Accuracy difference vs identity (pp)',title=f'Fresh holdout: 1536, selected G={G}\nDots: differences; stems: conservative lower bounds');ax.legend(fontsize=8)
ax=axes[1,2];xx=np.arange(3)
for R in [0,4]:
 vals=[d['selected_timing'][p][f'r{R}'] for p in ['preserve','native','warm']];x=xx+(R/4-.5)*.25
 ax.bar(x,[v['relative_change_pct'] for v in vals],width=.24,color=colors[R],alpha=.75,label=f'R={R}')
 for xi,v in zip(x,vals):lo,hi=v['block_bootstrap_relative95_pct'];ax.plot([xi,xi],[lo,hi],color='black');ax.plot([xi-.035,xi+.035],[lo,lo],color='black');ax.plot([xi-.035,xi+.035],[hi,hi],color='black')
ax.axhline(0,color='black',lw=.7);ax.set(xticks=xx,xticklabels=['Cold preserve','Cold native','Warm'],ylabel='TTFT change (%) — lower is faster',title=f'Selected G={G}: 4 paired blocks\nRepeated 16 cold / 32 warm questions');ax.legend()
fig.suptitle('Layer-specific Q/output readers with shared prefix KV — v2.5',fontsize=17)
(r/'figures').mkdir(exist_ok=True)
for ext in ['png','pdf']:fig.savefig(r/f'figures/v25_coverage_quality_latency.{ext}',dpi=170)
print('WROTE figures/v25_coverage_quality_latency.png and .pdf')
