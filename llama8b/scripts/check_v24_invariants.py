from pathlib import Path
import hashlib,json,sys
import torch
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'src'))
from contiguous_fuxian.sharedkv_v24.guard import check
check(r)
before=json.loads((r/'audit/v24_before.json').read_text())
changed=[p for p,h in before['files'].items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not changed,changed
original=[str(p) for p in Path('/home/panzihang/src/prism_pans').rglob('*') if p.is_file() and p.stat().st_mtime>before['time']]
assert not original,original
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
runs={}
for p in sorted((r/'results').glob('v24_*.extra.jsonl')):
 checks={'alias':True,'unique_sources':True,'physical_resolves':True,'private_tail':True,'buffer_bytes':True}
 rows=lines(p)
 assert rows,p
 for x in rows:
  ptr=[tuple(v) for v in x['prefix_storage_ptrs_by_layer']];sources=x['sources']
  checks['alias'] &= all(ptr[i]==ptr[s] for i,s in enumerate(sources))
  checks['unique_sources'] &= len(set(ptr))==len(set(sources))
  checks['physical_resolves'] &= x['physical_resolve_calls']==len(set(sources))
  checks['private_tail'] &= len(x['private_tail_lengths'])==28 and all(x['query_tokens']<=n<=x['query_tokens']+x['requested_max_tokens']+4 for n in x['private_tail_lengths'].values())
  checks['buffer_bytes'] &= x['request_prefix_buffer_bytes']==x['unique_selected_payload_bytes_fp16']
 assert all(checks.values()),(str(p),checks)
 runs[p.name]={'requests_including_warmup':len(rows),**checks}
selector={}
for p in sorted((r/'results').glob('v24_*.extra.jsonl.v21.jsonl')):
 rows=lines(p);assert all(v['selector_calls_prefill']==4 for v in rows)
 selector[p.name]={'all_four_selector_calls':True,'rows':len(rows)}
data=json.loads((r/'audit/v24_data_isolation.json').read_text())['query_hashes']
hashes={k:set(v.values()) for k,v in data.items()}
assert not hashes['dev768']&hashes['final768']
for split in ['dev768','final768']:
 assert not hashes[split]&(hashes['fit']|hashes['val']|hashes['test500']|hashes['dev'])
bundles={}
for p in sorted((r/'assets').glob('v24_*/trec.pt')):
 b=torch.load(p,map_location='cpu',weights_only=True);bans=set(b['calibration_query_hashes'])
 assert not bans&(hashes['dev768']|hashes['final768']|hashes['test500'])
 assert b['diagnostic_only'] and not b['end_to_end_validated']
 assert bans==hashes['fit']|hashes['val']
 bundles[p.parent.name]={'unique_sources':len(set(b['sources'])),'consumers':len(b['readers']),'diagnostic_only':True,'local_gate_passed':b.get('local_gate_passed'),'calibration_queries':len(bans),'reader_bytes_bf16':sum(v.numel()*2 for d in b['readers'].values() for v in d.values() if isinstance(v,torch.Tensor))}
private={}
for arm in ['identity','shared']:
 rows=lines(r/f'audit/v24_private_{arm}.jsonl')
 assert len({v['query_hash'] for v in rows})==8
 assert all(v['all_layers_private'] for v in rows)
 growth=max(v['private_lengths'][0]-v['query_tokens'] for v in rows);assert growth>1
 private[arm]={'queries':8,'decoder_calls':len(rows),'all_28_layers_distinct_K_and_V_storage':True,'max_private_growth':growth}
parity=lines(r/'audit/v24_shared_replay_parity.jsonl');assert len(parity)==8
assert all(x['max_token_error']<=.1 and x['max_mean_score_error']<=.02 for x in parity)
initial={}
for p in sorted((r/'audit').glob('v24_*_compute.json')):
 if p.stem.startswith(('v24_group_','v24_score_g26_','v24_nomargin_g26_')):
  d=json.loads(p.read_text());assert d['status']=='completed';initial[p.stem]=d['initial_reader_tensor_sha256']
assert len(initial)==9
assert all(v==next(iter(initial.values())) for v in initial.values())

# Recompute reported task decisions directly from saved full continuation scores.
scoring={}
for p in sorted((r/'results').glob('v24_*/scored_records.jsonl')):
 rows=lines(p)
 if not rows:continue
 for x in rows:
  scores=x['label_candidate_mean_logprobs'];tokens=x['label_candidate_token_logprobs']
  assert all(abs(scores[k]-sum(tokens[k])/len(tokens[k]))<1e-6 for k in scores),(p,x['uid'])
  assert x['prediction']==max(scores,key=scores.get),(p,x['uid'])
  assert x['correct']==(x['prediction']==x['answer']),(p,x['uid'])
 scoring[p.parent.name]={'requests':len(rows),'mean_candidate_scoring_prediction_and_correctness_consistent':True}


nsys=json.loads((r/'results/v24_nsight_summary.json').read_text())['variants']
expected={f'nsys_{p}_{a}' for p in ['native','preserve','warm'] for a in ['identity','shared']}
assert set(nsys)==expected,set(nsys)
assert all(x['n']==8 for x in nsys.values())
viz={}
for policy in ['native','warm']:
 for arm in ['identity','shared']:
  k=policy+'_'+arm;x=json.loads((r/f'results/v24_viz_{k}_summary.json').read_text())
  assert x['prefill_count']==8 and not x['overflow'],k
  viz[k]={'prefill_count':8,'overflow':False}

report={'all_passed':True,'profiler_completeness':{'nsight_variants':sorted(nsys),'requests_each':8,'viztracer':viz},'score_records_consistency':scoring,'prior_files_checked':len(before['files']),'changed_prior_files':changed,'original_files_modified':original,'physical_private_runtime':runs,'selector_calls':selector,'bundles':bundles,'actual_private_decode':private,'shared_scorer_replay_parity':{'n':8,'max_token_error':max(v['max_token_error'] for v in parity),'max_mean_score_error':max(v['max_mean_score_error'] for v in parity)},'all_nine_G26_initializations_identical':True,'final_dev_calibration_overlap':0}
(r/'audit/v24_final_invariants.json').write_text(json.dumps(report,indent=2));print('ALL INVARIANTS PASSED',len(runs),'runs',len(bundles),'bundles',len(before['files']),'protected files')
