from pathlib import Path
import json,hashlib
r=Path('/home/panzihang/src/prism_layerkv');root=r/'studies/v28_frozen'
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
m=json.loads((root/'manifest.json').read_text());checks=[];cold=0
for c in m['cells']:
 for arm in ['I','R4','R0']:
  output=root/'runs'/f"{c['id']}_quality_{arm}"
  rr=rows(output/'scored_records.jsonl')
  assert len(rr)==c['expected_n']
  extras=rows(Path(str(output)+'.extra.jsonl'))
  assert len(extras)==len(rr)
  for e in extras:
   assert len(e['private_tail_lengths'])==28
   assert all(v==e['query_tokens']+1 for v in e['private_tail_lengths'].values())
   pointers=e['prefix_storage_ptrs_by_layer']
   assert all(pointers[i]==pointers[s] for i,s in enumerate(e['sources']))
  if arm!='I':
   v25=rows(Path(str(output)+'.extra.jsonl.v25.jsonl'))
   assert len(v25)==len(rr)
   assert all(x['full_source_resolve_calls']==16 for x in v25)
   if arm=='R0':assert all(x['native_repair_resolve_calls']==0 and x['native_repair_payload_bytes']==0 for x in v25)
  checks.append(dict(cell=c['id'],arm=arm,n=len(rr),correct=sum(x['correct'] for x in rr),private_and_alias_checked=True))
 for cond in ['preserve','warm']:
  for b in range(6):
   for arm in ['I','R4','R0']:
    out=root/'runs'/f"{c['id']}_timing_{arm}_{cond}_b{b:02d}"
    assert len(rows(out/'scored_records.jsonl'))==16
    if cond=='preserve':
     cert=rows(Path(str(out)+'.cold.jsonl'))
     assert len(cert)==16
     assert all(x['status']=='completed' and x['page_after']['resident']==0 for x in cert)
     cold+=len(cert)
assert cold==4608
gates=[json.loads(p.read_text()) for p in (root/'gates').glob('*.json')]
assert len(gates)==16 and all(x['all_structural_gates_passed'] for x in gates)
(r/'audit/v28_results_audit.json').write_text(json.dumps(dict(main_quality=checks,cold_certified=cold,structural_gates=len(gates),passed=True),indent=2))
print('RESULTS_AUDIT_PASSED',cold,len(checks),len(gates))
