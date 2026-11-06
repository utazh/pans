
from v26_common import *
import random
from contiguous_fuxian.sharedkv_v26.study import build,sha256,read_bundle
while not all((ROOT/f'audit/v26_quality_gpu{g}_done.json').exists() for g in [2,3]):
 for name in ['v26_gpu2_sequence','v26_gpu3_sequence2']:
  if 'Traceback' in (ROOT/f'logs/{name}.log').read_text():raise RuntimeError(name+' failed; formal timing not started')
 time.sleep(10)
# Only this serial worker runs GPU requests after both quality workers finish.
spec=dict(repo=str(ROOT),python=PY,gpu=3,task='trec',model_path=MODEL,store_root=str(ROOT/'runtime_data/prefix_store'),selector_index_dir=str(ROOT/'assets/selector_index_k4_g32'),pcache_root=str(ROOT/'runtime_data/v26_pcache_gpu3'),adapter_dir=str(ROOT/'assets/v25_g16_r4_s24'),checkpoint_sha256=sha256(ROOT/'assets/v25_g16_r4_s24/trec.pt'),bundle_file=str(ROOT/'data/v26/p0/eval/trec.jsonl'),blocks=8,block_size=16,warmup=4,conditions=['preserve','warm'],budget=25,period=8,seed=2601,method_frozen=True,admission_gpu_parity_passed=True)
study=ROOT/'results/v26_primary_study';build(spec,study)
run('v26_primary_study',[PY,'-u','scripts/run_study_v26.py','--manifest',str(study/'study.json'),'--execute'],3)
cells=[('p0',t,b) for t in ['sst2','subj','trec','rte'] for b in [5,10,25,50]]+[('p1',t,25) for t in ['sst2','subj','trec','rte']]
manifest=[]
for i,(pref,task,budget) in enumerate(cells):
 tag=f'v26_{pref}_{task}_b{budget}'+('_canonical' if task=='subj' else '');frozen=json.loads((ROOT/f'audit/{tag}_frozen.json').read_text());adapter=Path(frozen['checkpoint']).parent
 assert sha256(adapter/f'{task}.pt')==frozen['sha256']
 evalsplit='canonical_eval' if task=='subj' else 'eval'
 rows=read_bundle(ROOT/f'data/v26/{pref}/{evalsplit}/{task}.jsonl',task)
 random.Random(2601).shuffle(rows);part=rows[:128];assert len(part)==128
 folder=ROOT/f'data/v26/{pref}/timing_{task}_b{budget}';folder.mkdir(exist_ok=False)
 with (folder/f'{task}.jsonl').open('x') as f:f.write(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in part))
 for cold in ['preserve',None]:
  for shared in ([False,True] if i%2==0 else [True,False]):
   name=tag+'_timing_'+(cold or 'warm')+'_'+('S' if shared else 'I')
   cmd=argv(name,task=task,budget=budget,gpu=3,shared=shared,admission='reuse-prefetch',n=128,warm=4,cold=cold,prefix=pref,split=folder.name,adapter=adapter)
   manifest.append(dict(name=name,prefix=pref,task=task,budget=budget,condition=cold or 'warm',shared=shared,checkpoint_sha256=frozen['sha256'],bundle_sha256=sha256(folder/f'{task}.jsonl'),command=cmd))
   with (ROOT/'audit/v26_matrix_timing_manifest.jsonl').open('a') as f:f.write(json.dumps(manifest[-1])+'\n')
   run(name,cmd,3)
with (ROOT/'audit/v26_formal_done.json').open('x') as f:json.dump(dict(status='completed',matrix_arms=len(manifest),primary_arms=64),f)
