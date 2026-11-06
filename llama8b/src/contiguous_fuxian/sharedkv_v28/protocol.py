from __future__ import annotations
import copy,json,itertools,random,sys
from pathlib import Path
from .common import *

def make_spec(repo,model,pcache,python=None):
 repo=Path(repo).resolve();cells=[]
 for task in TASKS:
  for b in BUDGETS:
   tag=f'v26_p0_{task}_b{b}'+('_canonical' if task=='subj' else '')
   special=task=='trec' and b==25
   cells.append(dict(id=f'{task}_b{b}',task=task,budget=b,expected_n=TASKS[task],
     r4=str(repo/('assets/v25_g16_r4_s24/trec.pt' if special else f'assets/{tag}/{task}.pt')),
     teacher_fit=str(repo/('runtime_data/v24_teacher_fit' if special else f'runtime_data/{tag}_teacher_fit')),
     teacher_val=str(repo/('runtime_data/v24_teacher_val' if special else f'runtime_data/{tag}_teacher_val')),
     eval_file=str(repo/f"data/v26/p0/{'canonical_full_eval' if task=='subj' else 'full_eval'}/{task}.jsonl"),
     validation_file=str(repo/f"data/v26/p0/{'canonical_val' if task=='subj' else 'val'}/{task}.jsonl")))
 return dict(format='pans-matrix-spec-v28',repo=str(repo),model_path=str(Path(model).resolve()),
    store_root=str(repo/'runtime_data/prefix_store'),selector_index_dir=str(repo/'assets/selector_index_k4_g32'),
    pcache_root=str(Path(pcache).resolve()),python=python or sys.executable,gpu='0',
    blocks=6,block_size=16,warmup=4,seed=2801,conditions=['preserve','warm'],
    ranks=[[256,384],[384,576]],kvsharer_g=[24,20,16],svd_device='cpu',search_device='cuda',
    cells=cells,paths_are_templates_verify_before_prepare=True,
    note='Frozen R4-trained -> R0-inference; no retraining. External bridges QUALITY ONLY.')

def assert_frozen_readers(parent,child):
 import torch
 for k in ('sources',*FIELDS,'calibration_query_hashes'):
  if parent[k]!=child[k]:raise ValueError('R0 intervention changed '+k)
 if child['native_policy']['blocks']!=0 or parent['native_policy']['blocks']!=4:raise ValueError('Expected R4 -> R0')
 if set(parent['readers'])!=set(child['readers']):raise ValueError('Reader keys changed')
 for lid,state in parent['readers'].items():
  if set(state)!=set(child['readers'][lid]):raise ValueError('Tensor keys changed')
  for k,t in state.items():
   c=child['readers'][lid][k]
   if not torch.equal(t,c) or t.dtype!=c.dtype:raise ValueError('Frozen reader tensor changed')

def prepare(spec,folder):
 from contiguous_fuxian.sharedkv_v24.records import load,assert_disjoint,assert_plan
 from contiguous_fuxian.sharedkv_v27.frozen_control import export_frozen_r0
 from contiguous_fuxian.sharedkv_v26.study import read_bundle
 from transformers import AutoTokenizer
 repo=Path(spec['repo']).resolve();check(repo);folder=Path(folder).resolve();folder.relative_to(repo)
 if folder.exists():raise FileExistsError(folder)
 if spec.get('paths_are_templates_verify_before_prepare',True):raise ValueError('Review all paths, then set paths_are_templates_verify_before_prepare=false')
 if len(set(spec['conditions']))!=len(spec['conditions']) or not spec['conditions'] or any(x not in ('warm','preserve','native') for x in spec['conditions']):raise ValueError('Invalid timing conditions')
 if spec.get('ranks') != [[256,384],[384,576]] or spec.get('kvsharer_g') != [24,20,16]:raise ValueError('Frozen primary baseline grid changed; preregister a separate study instead')
 for key in ('model_path','store_root','selector_index_dir','pcache_root'):
  if not Path(spec[key]).is_dir():raise FileNotFoundError(spec[key])
 nblocks=int(spec['blocks']);size=int(spec['block_size']);warm=int(spec['warmup'])
 if nblocks<3 or nblocks%3 or not 1<=warm<=size:raise ValueError('Blocks multiple of 3; warmup in [1,block_size]')
 expected={(t,b) for t in TASKS for b in BUDGETS}
 if {(c['task'],c['budget']) for c in spec['cells']}!=expected or len(spec['cells'])!=16:
  raise ValueError('Exactly four tasks x four budgets required')
 tok=AutoTokenizer.from_pretrained(spec['model_path'],local_files_only=True,trust_remote_code=False)
 frozen=[];allfiles={};prefix_by_task={};bundle_by_task={}
 for p in (Path(spec['model_path'])/'config.json',Path(spec['selector_index_dir'])/'manifest.json'):
  allfiles[str(p)]=sha(p)
 for c0 in spec['cells']:
  c=copy.deepcopy(c0);require_cell(c);r4=load_tensor(c['r4'])
  fit=load(c['teacher_fit'],c['task'],'fit');val=load(c['teacher_val'],c['task'],'validation');assert_disjoint(fit,val)
  header=next(iter(fit.values()));compare_identity(r4,header)
  if r4['keep_ratio']!=c['budget']/100 or r4['native_policy']['blocks']!=4:raise ValueError('Wrong cell checkpoint policy')
  for row in list(fit.values())+list(val.values()):assert_plan(row,r4['sources'])
  rows=read_bundle(c['eval_file'],c['task']);vr=read_bundle(c['validation_file'],c['task'])
  if len(vr)<16:raise ValueError('At least 16 distinct teacher-validation requests required for parity')
  if len(rows)!=c['expected_n']:raise ValueError('Full split count mismatch '+c['id'])
  if len(rows)<nblocks*size:raise ValueError('Not enough distinct timing queries')
  from contiguous_fuxian.sharedkv_v24.records import digest_ids as query_hash
  eval_hashes=[query_hash(tok(r['query_text'],add_special_tokens=False).input_ids) for r in rows]
  if set(eval_hashes)&(set(fit)|set(val)|set(r4['calibration_query_hashes'])):
   raise ValueError('Evaluation query intersects calibration bans')
  if len(set(eval_hashes))!=len(eval_hashes):raise ValueError('Duplicate model-tokenized eval queries')
  validation_hashes=[query_hash(tok(r['query_text'],add_special_tokens=False).input_ids) for r in vr]
  if not set(validation_hashes)<=set(val):raise ValueError('validation_file is not the explicit teacher validation split')
  if rows[0]['prefix_text']!=vr[0]['prefix_text']:raise ValueError('Validation/eval prefix differs')
  pids=tok(rows[0]['prefix_text'],add_special_tokens=False).input_ids
  # Native runtime verifies token_hash again; length and exact text checked here.
  if len(pids)!=header['prefix_tokens']:raise ValueError('Tokenized prefix length mismatch')
  ph=digest(rows[0]['prefix_text']);eh=sha(c['eval_file'])
  if c['task'] in prefix_by_task and (prefix_by_task[c['task']]!=ph or bundle_by_task[c['task']]!=eh):
   raise ValueError('Same task budgets must use the SAME evaluation file and prefix')
  prefix_by_task[c['task']]=ph;bundle_by_task[c['task']]=eh
  c.update(identity={k:header[k] for k in FIELDS},r4_sha256=sha(c['r4']),eval_query_hashes=eval_hashes)
  c['r0']=str(folder/'readers'/c['id']/(c['task']+'.pt'))
  frozen.append((c,r4,rows,vr))
  allfiles[str(Path(spec['store_root'])/c['task']/'metadata.json')]=sha(Path(spec['store_root'])/c['task']/'metadata.json')
  allfiles[c['r4']]=sha(c['r4']);allfiles[c['eval_file']]=eh;allfiles[c['validation_file']]=sha(c['validation_file'])
  allfiles.update(tree_files(Path(c['teacher_fit'])/c['task']));allfiles.update(tree_files(Path(c['teacher_val'])/c['task']))
 # All input validation completes before any artifact is created.
 folder.mkdir(parents=True);cells=[]
 for c,parent,rows,vr in frozen:
  child=export_frozen_r0(c['r4'],c['r0']);assert_frozen_readers(parent,child);c['r0_sha256']=sha(c['r0']);allfiles[c['r0']]=c['r0_sha256']
  for kind,rr in [('eval',rows),('parity',sorted(vr,key=lambda r:str(r['uid']))[:16])]:
   p=folder/'bundles'/c['id']/kind/(c['task']+'.jsonl');write_jsonl(p,rr);allfiles[str(p)]=sha(p)
  rng=random.Random(spec['seed']+list(TASKS).index(c['task']));ix=list(range(len(rows)));rng.shuffle(ix)
  for b in range(nblocks):
   rr=[rows[j] for j in ix[b*size:(b+1)*size]]
   p=folder/'bundles'/c['id']/f'b{b:02d}'/(c['task']+'.jsonl');write_jsonl(p,rr);allfiles[str(p)]=sha(p)
  cells.append(c)
 manifest=dict(format='pans-frozen-matrix-v28',base=BASE,spec=spec,cells=cells,input_sha256=allfiles,
               root=str(folder),training_steps=0,quality_sets_previously_observed=True)
 write_json(folder/'manifest.json',manifest);return manifest

def orders(blocks):
 if blocks%3 or blocks<3:raise ValueError('Use a multiple of three blocks')
 perms=list(itertools.permutations(('I','R4','R0')))
 if blocks%6==0:return [perms[i%6] for i in range(blocks)]
 return [('I','R4','R0'),('R4','R0','I'),('R0','I','R4')]*(blocks//3)

def command(m,c,arm,phase,*,block=None,condition='warm',representation=None,logits=None,selection=None):
 spec=m['spec'];root=Path(m['root']);repo=Path(spec['repo']);kind='eval' if block is None else f'b{block:02d}'
 if phase.startswith('parity'):kind='parity'
 key=f"{c['id']}_{phase}_{arm}"+(f'_{condition}_b{block:02d}' if block is not None else '')
 output=root/'runs'/key;shared=arm in ('R4','R0')
 args=[spec['python'],str(repo/'scripts/run_evidence_v26.py'),'--admission','reuse-prefetch',
       '--audit-jsonl',str(output)+'.v26.jsonl','--','--model-path',spec['model_path'],
       '--store-root',spec['store_root'],'--selector-index-dir',spec['selector_index_dir'],'--',
       '--pcache-root',spec['pcache_root']]
 if condition!='warm':
  if representation:raise ValueError('External native-loader bridge cannot be cold timed')
  if condition not in ('native','preserve'):raise ValueError('Unknown condition')
  args+=['--cold','ssd-cold','--read-advice',condition,'--owned-cache-root',spec['pcache_root'],'--cold-log',str(output)+'.cold.jsonl']
 args+=['--','--mode','shared' if shared else 'identity','--backend','vllm2','--metadata-cache','--layout','hybrid',
        '--stats',str(output)+'.extra.jsonl']
 if shared:args+=['--adapters',str(Path(c['r4'] if arm=='R4' else c['r0']).parent),'--allow-candidate','--packed-ops']
 if representation or phase.startswith('parity'):args+=['--diagnostic']
 if logits:args+=['--logits',str(logits)]
 if selection:args+=['--selection-plan',str(selection)]
 samples=(spec['block_size'] if block is not None else (16 if kind=='parity' else c['expected_n']))
 args+=['--','--task',c['task'],'--budget',str(c['budget']),'--period','8','--max-tokens','1',
        '--warmup',str(spec['warmup'] if block is not None else 0),'--samples',str(samples),
        '--profile','none','--bundle-dir',str(root/'bundles'/c['id']/kind),'--output',str(output)]
 if representation:
  if shared or block is not None:raise ValueError('Quality-only external bridge')
  args=[spec['python'],str(repo/'scripts/run_baseline_quality_v27.py'),'--representation',str(representation),'--',*args[2:]]
 return dict(id=key,argv=args,output=str(output),task=c['task'],budget=c['budget'],arm=arm,
             phase=phase,condition=condition,block=block,quality_only=representation is not None,
             logits_dir=str(logits) if logits else None,selection_dir=str(selection) if selection else None,
             representation=str(representation) if representation else None)
