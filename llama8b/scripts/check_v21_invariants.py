#!/usr/bin/env python3
import json,hashlib,statistics,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
checks={};bad=[]
for p in sorted((ROOT/'results').glob('v21_*.extra.jsonl')):
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 tests={'source_pointer_alias':True,'distinct_physical_sources':True,'resolve_count':True,'private_tail_bounded':True,'prefix_buffer_bytes':True}
 for r in rows:
  ptr=[tuple(x) for x in r['prefix_storage_ptrs_by_layer']];sources=r['sources']
  tests['source_pointer_alias'] &= all(ptr[i]==ptr[s] for i,s in enumerate(sources))
  tests['distinct_physical_sources'] &= len(set(ptr))==len(set(sources))
  tests['resolve_count'] &= r['physical_resolve_calls']==len(set(sources))
  tests['private_tail_bounded'] &= len(r['private_tail_lengths'])==28 and all(r['query_tokens']<=n<=r['query_tokens']+r['requested_max_tokens']+4 for n in r['private_tail_lengths'].values())
  tests['prefix_buffer_bytes'] &= r['request_prefix_buffer_bytes']==r['unique_selected_payload_bytes_fp16']
 checks[p.name]={'requests_including_warmup':len(rows),**tests}
 if not all(tests.values()):bad.append(p.name)
extra={}
for p in sorted((ROOT/'results').glob('v21_*.extra.jsonl.v21.jsonl')):
 rs=[json.loads(x) for x in p.read_text().splitlines()]
 extra[p.name]={'selector_calls_always_4':all(r['selector_calls_prefill']==4 for r in rs),'shadow_resolve_calls':sorted({r['shadow_resolve_calls'] for r in rs}),'packed_extra_bytes':sorted({r['packed_reader_extra_storage_bytes'] for r in rs})}
 if not extra[p.name]['selector_calls_always_4']:bad.append(p.name)
fixed={}
base={p.stem:p for p in (ROOT/'results/v21_fixed_shared_logits').glob('*.pt')}
for name in ('shadow','demand'):
 other={p.stem:p for p in (ROOT/f'results/v21_fixed_{name}_logits').glob('*.pt')}
 assert set(base)==set(other)
 logits=plans=labels=True;maxerr=0.
 for h in base:
  a=torch.load(base[h],map_location='cpu',weights_only=False);b=torch.load(other[h],map_location='cpu',weights_only=False)
  logits &= torch.equal(a['logits'],b['logits']);plans &= a['selected_by_layer']==b['selected_by_layer']
  labels &= a['label_token_logprobs']==b['label_token_logprobs'];maxerr=max(maxerr,float((a['logits'].float()-b['logits'].float()).abs().max()))
 fixed[name]={'n':len(base),'logits_bitwise_equal':logits,'selection_exact':plans,'label_token_logprobs_exact':labels,'max_logit_abs_diff':maxerr}
 if not (logits and plans and labels):bad.append('fixed_'+name)
# Validate token-level query disjointness using the exact runner tokenizer.
from transformers import AutoTokenizer
sys.path.insert(0,str(ROOT/'src'))
from contiguous_fuxian.sharedkv.io import query_hash
tokenizer=AutoTokenizer.from_pretrained('/data1/llm/Qwen/Qwen2.5-7B-Instruct',local_files_only=True)
sets={};uid_hashes={}
for name,path in [('old_fit','data/sharedkv_splits/fit'),('old_val','data/sharedkv_splits/val'),('reg','data/sharedkv_splits/test'),('fit','data/v21_splits/fit'),('val','data/v21_splits/val'),('holdout','data/v21_splits/holdout')]:
 rs=[json.loads(x) for x in (ROOT/path/'trec.jsonl').read_text().splitlines()]
 mapping={r['uid']:query_hash(tokenizer.encode(r['query_text'],add_special_tokens=False)) for r in rs}
 uid_hashes.update(mapping);sets[name]=set(mapping.values())
 assert len(sets[name])==len(rs),(name,'token duplicate')
(ROOT/'results/v21_query_hash_manifest.json').write_text(json.dumps(uid_hashes,indent=2))
for split,capture in [('reg','results/v21_reg_identity_logits'),('holdout','results/v21_hold_original_logits'),('fit','runtime_data/v21_teacher_fit/trec'),('val','runtime_data/v21_teacher_val/trec')]:
 assert sets[split]=={p.stem for p in (ROOT/capture).glob('*.pt')},('actual capture hash mismatch',split)
overlap={a+'__'+b:len(sets[a]&sets[b]) for i,a in enumerate(sets) for b in list(sets)[i+1:]}
if any(overlap.values()):bad.append('query_token_overlap')
report={'all_passed':not bad,'failures':bad,'physical_and_private':checks,'v21_runtime':extra,'fixed_mathematics':fixed,'tokenized_split_sizes':{k:len(v) for k,v in sets.items()},'tokenized_pairwise_overlap':overlap}
(ROOT/'results/v21_invariants.json').write_text(json.dumps(report,indent=2))
print('all_passed',not bad,'fixed',fixed,'sizes',report['tokenized_split_sizes'])
if bad:raise RuntimeError(bad)
