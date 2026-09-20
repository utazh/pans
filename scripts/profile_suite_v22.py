import json,subprocess,sys,os
from pathlib import Path
r=Path(__file__).resolve().parents[1]
nsys=r/'runtime_tools/nsight-2025.5.1/opt/nvidia/nsight-systems-cli/2025.5.1/target-linux-x64/nsys'
root=r/'profiles/v22';root.mkdir(parents=True,exist_ok=True)
for name in ('diag19_s200','map19_s200'):
 out=f'results/v22_profile_{name}'
 args=[sys.executable,'-u','scripts/profile_layerkv_v22.py','--profiler','nsys','--profile-output',str(root/f'nsys_{name}_metadata'),'--','--mode','shared','--adapters','assets/v22_'+name,'--allow-candidate','--metadata-cache','--packed-ops','--layout','hybrid','--diagnostic','--nvtx','--stats',out+'.extra.jsonl','--','--task','trec','--budget','25','--period','8','--samples','8','--warmup','4','--bundle-dir','data/v21_splits/holdout','--output',out]
 cmd=[str(nsys),'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output',str(root/f'nsys_{name}'),*args]
 print('RUN',cmd,flush=True)
 subprocess.run(cmd,cwd=r,check=True)
 subprocess.run([str(nsys),'export','--type=sqlite','--output',str(root/f'nsys_{name}.sqlite'),str(root/f'nsys_{name}.nsys-rep')],cwd=r,check=True)
name='map19_s200';out='results/v22_viz_'+name
subprocess.run([sys.executable,'-u','scripts/profile_layerkv_v22.py','--profiler','viztracer','--profile-output',str(root/'viz_map19'),'--','--mode','shared','--adapters','assets/v22_'+name,'--allow-candidate','--metadata-cache','--packed-ops','--layout','hybrid','--diagnostic','--nvtx','--stats',out+'.extra.jsonl','--','--task','trec','--budget','25','--period','8','--samples','8','--warmup','4','--bundle-dir','data/v21_splits/holdout','--output',out],cwd=r,check=True)
subprocess.run([sys.executable,'scripts/analyze_v22_nsight.py'],cwd=r,check=True)
subprocess.run([sys.executable,'scripts/analyze_v22_viz.py'],cwd=r,check=True)
print('PROFILING COMPLETE',flush=True)
