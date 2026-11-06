import json,hashlib,sys
from pathlib import Path
import torch
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'src'))
fail=[];runs={}
for p in sorted((r/'results').glob('v23_*.extra.jsonl')):
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 checks=dict(alias=True,unique_sources=True,physical_resolves=True,private_tail=True,buffer_bytes=True)
 for x in rows:
  ptr=[tuple(v) for v in x['prefix_storage_ptrs_by_layer']];sources=x['sources']
  checks['alias'] &= all(ptr[i]==ptr[s] for i,s in enumerate(sources))
  checks['unique_sources'] &= len(set(ptr))==len(set(sources))
  checks['physical_resolves'] &= x['physical_resolve_calls']==len(set(sources))
  checks['private_tail'] &= len(x['private_tail_lengths'])==28 and all(x['query_tokens']<=n<=x['query_tokens']+x['requested_max_tokens']+4 for n in x['private_tail_lengths'].values())
  checks['buffer_bytes'] &= x['request_prefix_buffer_bytes']==x['unique_selected_payload_bytes_fp16']
 runs[p.name]={'requests_including_warmup':len(rows),**checks}
 if not all(checks.values()):fail.append(p.name)
runtime={}
for p in sorted((r/'results').glob('v23_*.extra.jsonl.v21.jsonl')):
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 runtime[p.name]={'selector_calls_four':all(x['selector_calls_prefill']==4 for x in rows)}
 if not runtime[p.name]['selector_calls_four']:fail.append(p.name)
data=json.loads((r/'audit/v23_data_isolation.json').read_text())
fit=set(data['query_hashes']['fit'].values());val=set(data['query_hashes']['val'].values());test=set(data['query_hashes']['test500'].values())
bundles={}
for arm in ['joint','independent','query']:
 b=torch.load(r/f'assets/v23_{arm}/trec.pt',map_location='cpu',weights_only=True)
 assert set(b['calibration_query_hashes'])==fit|val and not (fit|val)&test
 assert b['sources']==list(range(25))+[26,26,26]
 assert b['diagnostic_only'] and not b['end_to_end_validated']
 bundles[arm]={'calibration_queries':len(b['calibration_query_hashes']),'local_gate_passed':b['local_gate_passed'],'diagnostic_only':True,'reader_bytes_bf16':sum(v.numel()*2 for rd in b['readers'].values() for v in rd.values() if isinstance(v,torch.Tensor))}
b=torch.load(r/'assets/v23_query/trec.pt',map_location='cpu',weights_only=True)
v=torch.load(r/'assets/v23_query_full_validation/trec.pt',map_location='cpu',weights_only=True)
assert all(torch.equal(t,v['readers'][lid][k]) for lid,d in b['readers'].items() for k,t in d.items() if isinstance(t,torch.Tensor))
private={}
for arm in ['identity','joint']:
 rows=[json.loads(x) for x in (r/f'audit/v23_private_{arm}.jsonl').read_text().splitlines()]
 assert len({x['query_hash'] for x in rows})==8
 assert all(x['all_layers_private'] for x in rows)
 assert max(x['private_lengths'][0]-x['query_tokens'] for x in rows)>1
 private[arm]={'queries':8,'decoder_calls':len(rows),'all_layers_have_distinct_K_and_V_storage':True,'max_private_growth':max(x['private_lengths'][0]-x['query_tokens'] for x in rows)}
before=json.loads((r/'audit/v23_before.json').read_text())
changed=[p for p,h in before['files'].items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not changed,changed
modified=[str(p) for p in Path('/home/panzihang/src/prism_pans').rglob('*') if p.is_file() and p.stat().st_mtime>before['time']]
assert not modified,modified
report={'all_passed':not fail,'failures':fail,'runtime_physical_private':runs,'actual_private_storage':private,'selector_audit':runtime,'bundles':bundles,'training_test_overlap':0,'query_full_validation_reader_unchanged':True,'prior_files_checked':len(before['files']),'changed_prior_files':changed,'original_files_modified_after_snapshot':modified}
(r/'audit/v23_final_invariants.json').write_text(json.dumps(report,indent=2))
print('INVARIANTS',not fail,'runtime files',len(runs),'prior files',len(before['files']))
assert not fail,fail
