import json,hashlib,sys
from pathlib import Path
import torch
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'src'))
from contiguous_fuxian.sharedkv_v22.matching import FORMAT
tests={};runtime={};failures=[]
for p in sorted((r/'results').glob('v22_*.extra.jsonl')):
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 checks=dict(alias=True,unique_sources=True,resolve_count=True,private_tail_native=True,prefix_bytes=True)
 for x in rows:
  ptr=[tuple(v) for v in x['prefix_storage_ptrs_by_layer']];sources=x['sources']
  checks['alias'] &= all(ptr[i]==ptr[s] for i,s in enumerate(sources))
  checks['unique_sources'] &= len(set(ptr))==len(set(sources))
  checks['resolve_count'] &= x['physical_resolve_calls']==len(set(sources))
  checks['private_tail_native'] &= len(x['private_tail_lengths'])==28 and all(x['query_tokens']<=n<=x['query_tokens']+x['requested_max_tokens']+4 for n in x['private_tail_lengths'].values())
  checks['prefix_bytes'] &= x['request_prefix_buffer_bytes']==x['unique_selected_payload_bytes_fp16']
 tests[p.name]={'requests_including_warmup':len(rows),**checks}
 if not all(checks.values()):failures.append(p.name)
for p in sorted((r/'results').glob('v22_*.extra.jsonl.v21.jsonl')):
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 runtime[p.name]={'selector_4':all(x['selector_calls_prefill']==4 for x in rows)}
 if not runtime[p.name]['selector_4']:failures.append(p.name)
routes={}
for p in sorted((r/'results').glob('v22_*.extra.jsonl.v22.jsonl')):
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 routes[p.name]={'no_prefix_reorder':all(not x['prefix_kv_reordered_online'] for x in rows),'resident_bytes':sorted({x['resident_head_route_bytes'] for x in rows}),'maps':rows[0]['head_maps']}
 if not routes[p.name]['no_prefix_reorder']:failures.append(p.name)
fit={p.stem for p in (r/'runtime_data/v21_teacher_fit/trec').glob('*.pt')}
val={p.stem for p in (r/'runtime_data/v21_teacher_val/trec').glob('*.pt')}
reg={p.stem for p in (r/'results/v22_reg_identity_logits').glob('*.pt')}
assert (len(fit),len(val),len(reg))==(64,64,175)
assert not (fit&val or fit&reg or val&reg)
bundles={}
for p in sorted((r/'assets').glob('v22_*/trec.pt')):
 b=torch.load(p,map_location='cpu',weights_only=True);a=b['assignment']
 ef=set(a['edge_fit_query_hashes']);sel=set(a['assignment_query_hashes'])
 assert len(ef)==48 and len(sel)==16 and ef|sel==fit and not ef&sel
 assert set(b['calibration_query_hashes'])==fit|val
 assert b['format']==FORMAT and b['diagnostic_only'] and not b['end_to_end_validated']
 bundles[p.parent.name]={'head_maps':b['head_maps'],'local_gate':b['local_gate_passed'],'diagnostic_only':b['diagnostic_only'],'calibration_hashes':len(b['calibration_query_hashes'])}
before=json.loads((r/'audit/v22_before.json').read_text())
changed=[p for p,h in before['files'].items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not changed,changed
mtime_changed=[str(p) for p in Path('/home/panzihang/src/prism_pans').rglob('*') if p.is_file() and p.stat().st_mtime>before['time']]
assert not mtime_changed,mtime_changed
report={'all_passed':not failures,'failures':failures,'physical_private':tests,'runtime':runtime,'routes':routes,'bundles':bundles,'fit':64,'assignment_split':[48,16],'val':64,'regression':175,'fresh_final_test':False,'calibration_regression_overlap':0,'prior_files_checked':len(before['files']),'changed_prior_files':changed,'original_files_modified_after_snapshot':mtime_changed}
(r/'results/v22_invariants.json').write_text(json.dumps(report,indent=2))
print('invariants',not failures,'runs',len(tests),'bundles',len(bundles),'prior_files',len(before['files']))
assert not failures,failures
