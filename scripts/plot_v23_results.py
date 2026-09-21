from pathlib import Path
import json,sys,importlib.util,subprocess
project=Path(__file__).resolve().parents[1]
deps=project/'runtime_deps_plot';sys.path.insert(0,str(deps))
if importlib.util.find_spec('matplotlib') is None:
 subprocess.run([sys.executable,'-m','pip','--isolated','install','--index-url','https://pypi.org/simple','--no-cache-dir','--target',str(deps),'--report',str(project/'audit/v23_plot_dependency_install.json'),'matplotlib'],check=True)
import importlib
importlib.invalidate_caches()
sys.path_importer_cache.pop(str(deps),None)
import os
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parents[1]/'runtime_deps/matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path(__file__).resolve().parents[1];(r/'audit/v23_plot_versions.json').write_text(json.dumps({'matplotlib':matplotlib.__version__,'numpy':np.__version__}));x=json.loads((r/'results/v23_final_analysis.json').read_text())
fig,ax=plt.subplots(2,2,figsize=(11,7.8),layout='constrained')
arms=['identity','independent','query','joint'];labels=['No sharing','Independent','Query only','Joint + suffix']
vals=[x['full500_warm']['variants'][a]['accuracy_pct'] for a in arms]
bars=ax[0,0].bar(labels,vals,color=['#667085','#94a3b8','#60a5fa','#0d9488'])
ax[0,0].bar_label(bars,fmt='%.1f%%');ax[0,0].set(ylim=(0,100),ylabel='Accuracy (%)',title='Complete TREC test set (500 queries)')
for panel,key,title in [(ax[0,1],'warm_paired_timing','Warm: 8 paired blocks (order sensitive)'),(ax[1,0],None,'SSD-cold: 10 paired blocks of 50 queries')]:
 d=x[key] if key else x['cold_full500']['paired_timing'];bs=d['per_block'];idx=np.arange(1,len(bs)+1)
 panel.plot(idx,[v['identity_ms'] for v in bs],'o-',label='No sharing',color='#667085')
 panel.plot(idx,[v['shared_ms'] for v in bs],'o-',label='Joint + suffix',color='#0d9488')
 panel.set(xlabel='Paired block',ylabel='Mean TTFT (ms)',title=title);panel.legend(frameon=False);panel.grid(alpha=.2)
 ci=d['block_bootstrap_change95_pct']
 panel.set_title(title+f"\nLatency {d['relative_latency_change_pct']:+.1f}%; block bootstrap 95% [{ci[0]:+.1f}, {ci[1]:+.1f}]%",fontsize=10)
v=x['full500_warm']['variants'];base=v['identity']['physical_payload_bytes'];shared=v['joint']['physical_payload_bytes']
payload=[base/2**20,shared/2**20];reader=[0,794624/2**20]
ax[1,1].bar(['No sharing','Joint + suffix'],payload,label='Unique selected prefix KV',color='#667085')
ax[1,1].bar(['No sharing','Joint + suffix'],reader,bottom=payload,label='Reader + packed parameters',color='#0d9488')
ax[1,1].set(ylabel='MiB',ylim=(0,85),xlabel='Persistent disk files are retained.',title='Per-request selected KV and reader storage\nPrefix KV -7.12%; including readers -5.99%')
ax[1,1].legend(frameon=False,fontsize=8,loc='upper right')
fig.suptitle('Qwen2.5-7B | TREC | 25% budget | Source 26 -> layers 25 and 27',fontsize=13)
out=r/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'v23_summary.png',dpi=180);fig.savefig(out/'v23_summary.pdf')
print(out/'v23_summary.png')
