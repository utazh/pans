
from v26_common import *
while not (ROOT/'audit/v26_formal_done.json').exists():
 if 'Traceback' in (ROOT/'logs/v26_formal_sequence.log').read_text():raise RuntimeError('Formal phase failed; diagnostics not started')
 time.sleep(10)
direct=ROOT/'assets/v26_direct_g16_r4'
run('v26_export_direct',[PY,'-u','scripts/export_direct_v26.py','--input',str(ROOT/'assets/v25_g16_r4_s24/trec.pt'),'--output',str(direct/'trec.pt')],3)
name='v26_mechanism_direct'
run(name,argv(name,shared=True,admission='reuse-prefetch',n=500,warm=4,diagnostic=True,split='full_eval',adapter=direct),3)
name='v26_mechanism_proxy';base=argv(name,shared=True,admission='reuse-prefetch',proxy=True,n=500,warm=0,diagnostic=True,split='full_eval')
run(name,[PY,'-u','scripts/audit_proxy_v26.py','--rounding-audit',str(ROOT/'audit/v26_proxy_rounding.json'),'--',*base[3:]],3)
nsys=ROOT/'runtime_tools/nsight-2025.5.1/opt/nvidia/nsight-systems-cli/2025.5.1/target-linux-x64/nsys'
for profiler in ['nsys','viztracer']:
 for shared in [False,True]:
  for admission in ['original','reuse-prefetch']:
   name=f'v26_profile_{profiler}_'+('S' if shared else 'I')+'_'+admission
   target=ROOT/f'profiles/v26/{name}'
   base=argv(name,shared=shared,admission=admission,n=8,warm=4,cold='preserve',diagnostic=True)
   ix=base.index('--mode');base.insert(ix,'--nvtx')
   cmd=[PY,'-u','scripts/profile_layerkv_v26.py','--profiler',profiler,'--profile-output',str(target),'--',*base[3:]]
   if profiler=='nsys':cmd=[str(nsys),'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output',str(target),*cmd]
   run(name,cmd,3)
   if profiler=='nsys':run(name+'_export',[str(nsys),'export','--type','sqlite','--output',str(target)+'.sqlite',str(target)+'.nsys-rep'],3)
with (ROOT/'audit/v26_diagnostics_done.json').open('x') as f:json.dump(dict(status='completed',profiled_requests_per_arm=8,profiler_conditions=['SSD-cold/preserve']),f)
