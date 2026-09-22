
from pathlib import Path
import hashlib,json,sys,torch
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'src'))
from contiguous_fuxian.sharedkv_v25.guard import check
check(r)
before=json.loads((r/'audit/v25_before.json').read_text())
changed=[p for p,h in before['files'].items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not changed,changed
original=[str(p) for p in Path('/home/panzihang/src/prism_pans').rglob('*') if p.is_file() and p.stat().st_mtime>before['time']]
assert not original,original
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
physical={}
for p in sorted((r/'results').glob('v25_*.extra.jsonl.v25.jsonl')):
 z=lines(p);assert z,p
 for x in z:
  assert x['full_source_buffer_bytes']==x['shared_base_payload_bytes']
  assert x['native_repair_buffer_bytes']==x['native_repair_payload_bytes']
  assert x['total_selected_payload_bytes']==x['full_source_buffer_bytes']+x['native_repair_buffer_bytes']
  assert x['full_source_resolve_calls']==x['full_selected_source_count']
  assert x['native_repair_resolve_calls']==x['partial_native_layer_count']
  assert x['total_resolve_calls']==x['full_source_resolve_calls']+x['native_repair_resolve_calls']
  base={tuple(v) for v in x['shared_base_storage_ptrs'].values()}
  native={tuple(v) for v in x['native_repair_storage_ptrs'].values()}
  assert len(base)==x['full_selected_source_count'] and len(native)==x['partial_native_layer_count'] and not base&native
  assert not x['quality_certified']
 physical[p.name]={'requests_including_warmup':len(z),'base_plus_repair_accounting_verified':True,'distinct_native_buffers_and_shared_sources':True}
scoring={}
for p in sorted((r/'results').glob('v25_*/scored_records.jsonl')):
 z=lines(p)
 if not z:continue
 assert len({x['uid'] for x in z})==len(z)
 for x in z:
  scores=x['label_candidate_mean_logprobs'];tokens=x['label_candidate_token_logprobs']
  assert all(abs(scores[k]-sum(tokens[k])/len(tokens[k]))<1e-6 for k in scores)
  assert x['prediction']==max(scores,key=scores.get) and x['correct']==(x['prediction']==x['answer'])
 scoring[p.parent.name]={'n':len(z),'all_task_scores_consistent':True}
h=json.loads((r/'audit/v25_data_isolation.json').read_text())['query_hashes'];sets={k:set(v.values()) for k,v in h.items()}
assert not sets['final1536']&(sets['fit']|sets['val']|sets['explore768']|sets['test500'])
bundles={}
from contiguous_fuxian.sharedkv_v25.policy import FORMAT
for p in sorted((r/'assets').glob('v25_*/trec.pt')):
 b=torch.load(p,map_location='cpu',weights_only=True)
 assert b['format']==FORMAT and b['diagnostic_only'] and not b['end_to_end_validated']
 bans=set(b['calibration_query_hashes']);assert bans==sets['fit']|sets['val']
 assert not bans&(sets['final1536']|sets['explore768']|sets['test500'])
 assert len(b['readers'])==28-len(set(b['sources']))
 bundles[p.parent.name]={'G':len(set(b['sources'])),'R':b['native_policy']['blocks'],'consumers':len(b['readers']),'calibration_queries':len(bans),'diagnostic_only':True}
freeze=json.loads((r/'audit/v25_frozen_selection.json').read_text());G=freeze['chosen_G']
for name,sha in freeze['checkpoint_sha256'].items():assert hashlib.sha256((r/f'assets/v25_{name}/trec.pt').read_bytes()).hexdigest()==sha
initials={}
for g in [26,24,22,20,16,14]:
 paths=[r/f'audit/v25_g{g}_r{R}_s24_compute.json' for R in [0,4]]
 hashes=[json.loads(p.read_text())['initial_reader_tensor_sha256'] for p in paths]
 assert hashes[0]==hashes[1],g
 initials[str(g)]={'R0_R4_exact_same_initialization':True}
for R in [0,4]:
 hs=[json.loads((r/f'audit/v25_g{G}_r{R}_s{s}_compute.json').read_text())['initial_reader_tensor_sha256'] for s in [24,25,26]]
 assert hs[0]==hs[1]==hs[2]
contracts={}
for arm in ['identity','r0','r4']:
 z=lines(r/f'audit/v25_contract_{arm}.jsonl');assert len(z)==8 and len({x['query_hash'] for x in z})==8
 assert all(x['all_28_private'] and x['support_partition_exact'] and x['native_values_equal_stored_layer'] and x['all_backend_prefixes_source_views'] and x['cache_admission_only_native_repairs'] for x in z)
 assert max(x['max_private_growth'] for x in z)>1
 contracts[arm]={'n':8,'max_private_growth':max(x['max_private_growth'] for x in z),'decoder_calls':sum(x['decoder_calls'] for x in z),'all_passed':True}
preflight=lines(r/'audit/v25_contract_preflight.jsonl');assert len(preflight)==1
assert preflight[0]['native_values_checked']==list(range(1,28,2)) and preflight[0]['max_private_growth']==15
assert all(preflight[0][k] for k in ['all_28_private','support_partition_exact','native_values_equal_stored_layer','all_backend_prefixes_source_views','cache_admission_only_native_repairs'])
parity={}
for R in [0,4]:
 z=lines(r/f'audit/v25_final_parity_r{R}.jsonl');assert len(z)==8
 assert all(x['max_token_error']<=.10 and x['max_mean_score_error']<=.02 for x in z)
 parity[str(R)]={'n':8,'max_token_error':max(x['max_token_error'] for x in z),'max_mean_score_error':max(x['max_mean_score_error'] for x in z)}
nsys=json.loads((r/'results/v25_nsight_summary.json').read_text())['variants']
assert set(nsys)=={f'nsys_{p}_{a}' for p in ['native','preserve','warm'] for a in ['identity','shared']}
assert all(x['n']==8 for x in nsys.values())
viz={}
for policy in ['native','warm']:
 for arm in ['identity','shared']:
  key=policy+'_'+arm;x=json.loads((r/f'results/v25_viz_{key}_summary.json').read_text())
  assert x['prefill_count']==8 and not x['overflow'],key
  viz[key]={'requests':8,'overflow':False}
selector={}
for p in sorted((r/'results').glob('v25_*.extra.jsonl.v21.jsonl')):
 z=lines(p);assert all(x['selector_calls_prefill']==4 for x in z)
 selector[p.name]={'n':len(z),'four_P8_selector_calls':True}
report={'all_passed':True,'prior_files_checked':len(before['files']),'changed_prior_files':changed,'original_files_modified':original,'physical_runtime':physical,'scoring':scoring,'bundles':bundles,'initializations':initials,'actual_contracts':contracts,'widest_G14_R4_preflight':preflight[0],'shared_scorer_replay_parity':parity,'profiler_completeness':{'nsight_variants':sorted(nsys),'viztracer':viz},'selector':selector,'final_no_calibration_overlap':True,'frozen_checkpoints_unchanged':True}
(r/'audit/v25_final_invariants.json').write_text(json.dumps(report,indent=2))
print('ALL V25 INVARIANTS PASSED',len(physical),'runtime arms',len(bundles),'bundles',len(before['files']),'protected files')
