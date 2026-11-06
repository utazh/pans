"""Analyze predeclared v24 comparisons; never choose checkpoints from final/test."""
from pathlib import Path
import json,sys,statistics,csv
import numpy as np
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'scripts'))
from audit_quality_v24 import paired_report
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def rows(name,n=None):
 x=lines(r/f'results/{name}/scored_records.jsonl')
 if n is not None:assert len(x)==n,(name,len(x))
 assert len({v['uid'] for v in x})==len(x)
 return x
def basic(x):
 return {'n':len(x),'correct':sum(v['correct'] for v in x),'accuracy_pct':100*statistics.mean(v['correct'] for v in x)}
def compare(a,b):
 aa={v['uid']:v for v in a};bb={v['uid']:v for v in b};d=paired_report(aa,bb)
 d['prediction_agreement']=statistics.mean(aa[k]['prediction']==bb[k]['prediction'] for k in aa)
 d['selection_plan_exact_agreement']=statistics.mean(aa[k]['layer_token_selection_sha256']==bb[k]['layer_token_selection_sha256'] for k in aa)
 paired=[]
 for uid in aa:
  x,y=aa[uid],bb[uid];answer=x['answer']
  def margin(q):
   s=q['label_candidate_mean_logprobs'];return s[answer]-max(v for k,v in s.items() if k!=answer)
  paired.append({'uid':uid,'baseline_correct':x['correct'],'student_correct':y['correct'],'baseline_prediction':x['prediction'],'student_prediction':y['prediction'],'gold_margin_change':margin(y)-margin(x)})
 d['mean_gold_margin_change']=statistics.mean(v['gold_margin_change'] for v in paired)
 fidelity=[]
 for uid in aa:
  x,y=aa[uid],bb[uid];labels=list(x['label_candidate_mean_logprobs'])
  ts=np.array([x['label_candidate_mean_logprobs'][k] for k in labels],dtype=float);ss=np.array([y['label_candidate_mean_logprobs'][k] for k in labels],dtype=float)
  def logsm(z):
   w=z-z.max();return w-np.log(np.exp(w).sum())
  tp,sp=logsm(ts),logsm(ss);ct,cs=ts-ts.mean(),ss-ss.mean()
  token=np.mean([np.mean((np.array(x['label_candidate_token_logprobs'][k])-np.array(y['label_candidate_token_logprobs'][k]))**2) for k in labels])
  order=np.sort(ts);margin=order[-1]-order[-2];error=np.max(np.abs(cs-ct))
  fidelity.append({'candidate_kl':float(np.sum(np.exp(tp)*(tp-sp))),'centered_score_mse':float(np.mean((ct-cs)**2)),'per_label_mean_token_mse':float(token),'sufficient_decision_preservation_bound':bool(margin>2*error)})
 d['online_score_fidelity']={k:statistics.mean(v[k] for v in fidelity) for k in fidelity[0]}
 d['online_score_fidelity']['scope']='Against paired online identity scores; softmax of mean candidate logprob is an auxiliary distribution, not full-vocabulary KL or calibrated class probability.'
 d['flips']=[v for v in paired if v['baseline_prediction']!=v['student_prediction']]
 return d
def timing(blocks):
 bm=np.array([np.mean([v['ttft_ms'] for v in a]) for a,b in blocks])
 sm=np.array([np.mean([v['ttft_ms'] for v in b]) for a,b in blocks])
 for a,b in blocks:assert [v['uid'] for v in a]==[v['uid'] for v in b]
 rng=np.random.default_rng(24);ix=rng.integers(0,len(blocks),(10000,len(blocks)))
 delta=(sm-bm)[ix].mean(1);rel=100*(sm[ix].mean(1)/bm[ix].mean(1)-1)
 return {'paired_blocks':len(blocks),'distinct_query_ids':len({v['uid'] for a,b in blocks for v in a}),'baseline_mean_ms':float(bm.mean()),'shared_mean_ms':float(sm.mean()),'relative_change_pct':float(100*(sm.mean()/bm.mean()-1)),'block_bootstrap_delta95_ms':np.quantile(delta,[.025,.975]).tolist(),'block_bootstrap_relative95_pct':np.quantile(rel,[.025,.975]).tolist(),'per_block':[{'identity_ms':float(a),'shared_ms':float(b),'delta_ms':float(b-a)} for a,b in zip(bm,sm)],'limitation':'Four repeated blocks on the same 32 queries; bootstrap across blocks is exploratory, not 128 independent query samples.'}
out={'protocol':json.loads((r/'audit/v24_preregistered_plan.json').read_text()),'frozen_selection':json.loads((r/'audit/v24_frozen_selection.json').read_text()),'scope':'Frozen Qwen2.5-7B, TREC task-conditioned readers, prefix4940, exact25% budget, onlineP8, private query/decode KV on all28layers. Quality-run TTFT excluded because own jobs ran concurrently.'}
quality={}
for split,n in [('dev',768),('final768',768),('test500',500)]:
 prefix=f'v24_{split}_';base=rows(prefix+'identity',n);variants={}
 for p in sorted((r/'results').glob(prefix+'*/scored_records.jsonl')):
  name=p.parent.name;v=rows(name,n);variants[name.removeprefix(prefix)]={'basic':basic(v),'paired_vs_identity':compare(base,v)}
 quality[split]=variants
out['quality']=quality
out['timing']={};out['cold_certification']={}
for policy in ['warm','native','preserve']:
 blocks=[(rows(f'v24_time_{policy}_b{i}_identity',32),rows(f'v24_time_{policy}_b{i}_shared',32)) for i in range(4)]
 out['timing'][policy]={'all':timing(blocks),'baseline_first':timing([blocks[i] for i in [0,3]]),'shared_first':timing([blocks[i] for i in [1,2]])}
 if policy!='warm':
  audit={}
  for arm in ['identity','shared']:
   z=[v for i in range(4) for v in lines(r/f'results/v24_time_{policy}_b{i}_{arm}.cold.jsonl')]
   assert len(z)==128
   assert all(v['status']=='completed' and v['page_after']['resident']==0 and v['application_payload_empty'] and v['gpu_mib_after']==v['cpu_mib_after']==0 for v in z)
   assert all(v['io_first_decoder']['write_bytes']==0 for v in z)
   audit[arm]={'requests':len(z),'all_cold_certified':True,'mean_setup_ms_excluded':statistics.mean(v['setup_ms_outside_ttft'] for v in z),'mean_process_read_bytes_first_decoder':statistics.mean(v['io_first_decoder']['read_bytes'] for v in z),'mean_major_faults_first_decoder':statistics.mean(v['io_first_decoder']['major_faults'] for v in z)}
  out['cold_certification'][policy]=audit
out['formal_runtime_counters']={}
keys=['selector_compute_ms','selector_load_ms','prefetch_wait_ms','cache_update_ms','prefetch_gpu_source_fraction','prefetch_cpu_source_fraction','prefetch_disk_source_fraction']
for policy in ['warm','native','preserve']:
 for arm in ['identity','shared']:
  z=[v for i in range(4) for v in rows(f'v24_time_{policy}_b{i}_{arm}',32)]
  out['formal_runtime_counters'][policy+'_'+arm]={k:statistics.mean(v[k] for v in z) for k in keys}
out['formal_runtime_counter_scope']='Existing runtime counters from formal rows. Host timings may include waits and overlap; they are not mutually exclusive GPU-kernel or critical-path durations.'
out['quality_consistency_across_timing']={}
chosen=out['frozen_selection']['chosen_G']
for arm in ['identity','shared']:
 reference=rows('v24_test500_'+('identity' if arm=='identity' else f'score_g{chosen}_s24'),500)
 ref={v['uid']:v for v in reference}
 checks={}
 for policy in ['warm','native','preserve']:
  z=[v for i in range(4) for v in rows(f'v24_time_{policy}_b{i}_{arm}',32)]
  checks[policy]={'requests':len(z),'prediction_disagreements_vs_frozen_full500':sum(v['prediction']!=ref[v['uid']]['prediction'] for v in z),'selection_hash_disagreements_vs_frozen_full500':sum(v['layer_token_selection_sha256']!=ref[v['uid']]['layer_token_selection_sha256'] for v in z)}
 out['quality_consistency_across_timing'][arm]=checks
confirmation=[(rows(f'v24_confirm_warm_b{i}_identity',32),rows(f'v24_confirm_warm_b{i}_shared',32)) for i in range(4)]
out['warm_fresh_process_confirmation']={'all':timing(confirmation),'baseline_first':timing([confirmation[i] for i in [0,3]]),'shared_first':timing([confirmation[i] for i in [1,2]]),'scope':'Post hoc lifecycle confirmation after retained-mapping cold rejection. Does not replace original four warm blocks.'}
out['training']={}
for p in sorted((r/'audit').glob('v24_*_compute.json')):
 d=json.loads(p.read_text());name=p.name.removesuffix('_compute.json')
 if name.startswith('v24_smoke'):continue
 rp=r/f'assets/{name}/trec.pt.report.json';report=json.loads(rp.read_text())
 out['training'][name]={**d,'best_step':report['best_step'],'final_validation':{k:v for k,v in report.get('final',report.get('report',{})).items() if k not in ['query_diagnostics','per_request']},'trainer_training_seconds':report.get('training_seconds_including_validation')}
out['resource']={}
for arm in ['identity','shared']:
 x=lines(r/f'results/v24_time_native_b0_{arm}.extra.jsonl')[8:];assert len(x)==32
 out['resource'][arm]={k:statistics.mean(v[k] for v in x) for k in ['physical_resolve_calls','logical_selected_payload_bytes_fp16','unique_selected_payload_bytes_fp16','request_prefix_buffer_bytes','resident_reader_bytes']}
for arm in ['identity','shared']:
 x=lines(r/f'results/v24_time_native_b0_{arm}.extra.jsonl.v21.jsonl')[8:];assert len(x)==32
 out['resource'][arm]['packed_reader_extra_storage_bytes']=statistics.mean(v['packed_reader_extra_storage_bytes'] for v in x)
base=out['resource']['identity']['request_prefix_buffer_bytes']
shared=out['resource']['shared']
out['resource']['selected_payload_reduction_pct']=100*(1-shared['request_prefix_buffer_bytes']/base)
out['resource']['selected_buffer_plus_readers_net_reduction_pct']=100*(1-(shared['request_prefix_buffer_bytes']+shared['resident_reader_bytes']+shared['packed_reader_extra_storage_bytes'])/base)
out['resource']['scope']='Selected request prefix buffers plus reader resident/packed copies only; not total model GPU memory or persisted disk footprint.'
out['test500_strict495']={}
for p in sorted((r/'results').glob('v24_test500_*/scored_records.jsonl')):
 x=[v for v in rows(p.parent.name,500) if v['uid'] not in {'trec-0','trec-32','trec-33','trec-34','trec-35'}]
 assert len(x)==495
 out['test500_strict495'][p.parent.name]=basic(x)
out['restore_diagnostic']={}
for policy in ['native','preserve']:
 for arm in ['identity','shared']:
  p=r/f'audit/v24_restore_{policy}_{arm}.jsonl'
  if not p.exists():continue
  z=lines(p)[4:];assert len(z)==8
  out['restore_diagnostic'][policy+'_'+arm]={'n':8,'prefetch':{k:statistics.mean(v['prefetch'][k] for v in z) for k in z[0]['prefetch']},'promotion':{k:statistics.mean(v['promotion'][k] for v in z) for k in z[0]['promotion']},'mean_files_seen_in_both_paths':statistics.mean(v['files_seen_in_both_paths'] for v in z),'mean_declared_overlap_file_bytes':statistics.mean(v['declared_overlap_file_bytes'] for v in z),'formal_timing_eligible':False,'byte_scope':'Read attempts times declared file size; neither hardware H2D nor guaranteed SSD traffic.'}
for key,path in [('nsight','results/v24_nsight_summary.json'),('invariants','audit/v24_final_invariants.json')]:
 if (r/path).exists():out[key]=json.loads((r/path).read_text())
for arm in ['identity','shared']:
 p=r/f'results/v24_viz_native_{arm}_summary.json'
 if p.exists():out['viz_'+arm]=json.loads(p.read_text())
for arm in ['identity','shared']:
 p=r/f'results/v24_viz_warm_{arm}_summary.json'
 if p.exists():out['viz_warm_'+arm]=json.loads(p.read_text())
(r/'results/v24_final_analysis.json').write_text(json.dumps(out,indent=2))
with (r/'results/v24_quality_table.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['split','variant','n','correct','accuracy_pct','harmful','beneficial','delta_pp','lower_pp','gate_pass'])
 for split,arms in quality.items():
  for arm,d in arms.items():
   b,p=d['basic'],d['paired_vs_identity'];w.writerow([split,arm,b['n'],b['correct'],b['accuracy_pct'],p['harmful'],p['beneficial'],p['net_delta_pp'],p['conservative_one_sided_lower_pp'],p['conservative_bound_passes']])
print(json.dumps({'quality':{s:{a:d['basic'] for a,d in z.items()} for s,z in quality.items()},'timing':out['timing']},indent=2))
