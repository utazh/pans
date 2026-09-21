#!/usr/bin/env python3
"""Fixed-plan v23 analysis; no checkpoint selection from test results."""
from pathlib import Path
import json,statistics,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from contiguous_fuxian.sharedkv_v21.audit import paired_quality
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
manifest=json.loads((ROOT/'audit/v23_data_isolation.json').read_text())['query_hashes']
hashes={k:v for group in manifest.values() for k,v in group.items()}
def records(name,n):
 rs=lines(ROOT/f'results/{name}/scored_records.jsonl');assert len(rs)==n,(name,len(rs),n)
 for row in rs:row['query_hash']=hashes[row['uid']]
 assert len({r['uid'] for r in rs})==n
 return rs
def basic(rs):
 return {'n':len(rs),'correct':sum(bool(r['correct']) for r in rs),'accuracy_pct':100*statistics.mean(r['correct'] for r in rs),
         'mean_ttft_ms':statistics.mean(r['ttft_ms'] for r in rs),'median_ttft_ms':statistics.median(r['ttft_ms'] for r in rs),
         'p95_ttft_ms':float(np.quantile([r['ttft_ms'] for r in rs],.95)),
         'physical_payload_bytes':statistics.mean(r['physical_prefetch_kv_bytes'] for r in rs),
         'logical_payload_bytes':statistics.mean(r['selected_kv_bytes'] for r in rs)}
def pair(b,s):
 result=paired_quality(b,s,bootstrap=5000,seed=23)
 bi={r['uid']:r for r in b};si={r['uid']:r for r in s};rows=[]
 score_deltas={label:[] for label in b[0]['label_candidate_mean_logprobs']}
 for uid in bi:
  x,y=bi[uid],si[uid];label=x['answer']
  def margin(r):
   scores=r['label_candidate_mean_logprobs']
   return scores[label]-max(v for k,v in scores.items() if k!=label)
  for k in score_deltas:score_deltas[k].append(y['label_candidate_mean_logprobs'][k]-x['label_candidate_mean_logprobs'][k])
  rows.append({'uid':uid,'answer':label,'baseline_prediction':x['prediction'],'student_prediction':y['prediction'],'baseline_gold_margin':margin(x),'student_gold_margin':margin(y),'harmful_flip':bool(x['correct'] and not y['correct']),'beneficial_flip':bool(y['correct'] and not x['correct'])})
 result['selection_plan_exact_agreement']=statistics.mean(bi[k]['layer_token_selection_sha256']==si[k]['layer_token_selection_sha256'] for k in bi)
 result['prediction_agreement']=statistics.mean(bi[k]['prediction']==si[k]['prediction'] for k in bi)
 result['mean_label_score_delta']={k:statistics.mean(v) for k,v in score_deltas.items()}
 result['mean_gold_margin_delta']=statistics.mean(r['student_gold_margin']-r['baseline_gold_margin'] for r in rows)
 result['paired_label_margins']=rows
 return result
def timing(blocks):
 bm=np.array([statistics.mean(r['ttft_ms'] for r in b) for b,s in blocks])
 sm=np.array([statistics.mean(r['ttft_ms'] for r in s) for b,s in blocks])
 for b,s in blocks:assert [r['uid'] for r in b]==[r['uid'] for r in s]
 rng=np.random.default_rng(23);ix=rng.integers(0,len(blocks),(10000,len(blocks)))
 delta=(sm-bm)[ix].mean(1);rel=100*(sm[ix].mean(1)/bm[ix].mean(1)-1)
 return {'blocks':len(blocks),'queries_per_block':[len(b) for b,s in blocks],'baseline_mean_ms':float(bm.mean()),'shared_mean_ms':float(sm.mean()),'delta_ms':float((sm-bm).mean()),'relative_latency_change_pct':float(100*(sm.mean()/bm.mean()-1)),'block_bootstrap_delta95_ms':np.quantile(delta,[.025,.975]).tolist(),'block_bootstrap_change95_pct':np.quantile(rel,[.025,.975]).tolist(),'per_block':[{'identity_ms':float(a),'shared_ms':float(b),'delta_ms':float(b-a)} for a,b in zip(bm,sm)]}
out={'protocol':json.loads((ROOT/'audit/v23_preregistered_plan.json').read_text()),
 'scope':'Qwen2.5-7B-Instruct, fixed TREC 4940-token prefix, 25% exact block budget, online P8; historically observed official test500. Adapter calibration uses training-only fit64/val64. Cold setup excluded from TTFT; profiler excluded.'}
for split,n,prefix in [('dev',128,'v23_dev_'),('full500_warm',500,'v23_full_')]:
 rows={arm:records(prefix+arm,n) for arm in ('identity','joint','independent','query')}
 out[split]={'variants':{k:basic(v) for k,v in rows.items()},
 'quality_vs_identity':{k:pair(rows['identity'],v) for k,v in rows.items() if k!='identity'},
 'joint_vs_independent':pair(rows['independent'],rows['joint']),
 'joint_vs_query_only':pair(rows['query'],rows['joint'])}
 if n==500:
  out[split]['strict495']={k:basic([r for r in v if r['uid'] not in {'trec-0','trec-32','trec-33','trec-34','trec-35'}]) for k,v in rows.items()}
warmblocks=[(records(f'v23_time_b{b}_identity',32),records(f'v23_time_b{b}_joint',32)) for b in range(8)]
out['warm_paired_timing']=timing(warmblocks)
out['warm_order_sensitivity']={
 'baseline_first':timing([v for i,v in enumerate(warmblocks) if i%4 in (0,3)]),
 'shared_first':timing([v for i,v in enumerate(warmblocks) if i%4 not in (0,3)])}
out['warm_timing_first4']=timing(warmblocks[:4]);out['warm_timing_confirm4']=timing(warmblocks[4:])
coldblocks=[(records(f'v23_cold_b{b:02}_identity',50),records(f'v23_cold_b{b:02}_joint',50)) for b in range(10)]
out['cold_order_sensitivity']={
 'baseline_first':timing([v for i,v in enumerate(coldblocks) if i%4 in (0,3)]),
 'shared_first':timing([v for i,v in enumerate(coldblocks) if i%4 not in (0,3)])}
coldrows={arm:[r for block in coldblocks for r in block[i]] for i,arm in enumerate(['identity','joint'])}
for arm,rs in coldrows.items():assert len({r['uid'] for r in rs})==500
out['cold_full500']={'variants':{a:basic(rs) for a,rs in coldrows.items()},'paired_timing':timing(coldblocks),'quality':pair(coldrows['identity'],coldrows['joint'])}
out['cold_audit']={}
for arm in ('identity','joint'):
 rs=[r for b in range(10) for r in lines(ROOT/f'results/v23_cold_b{b:02}_{arm}.cold.jsonl')]
 assert len(rs)==500 and len({r['query_hash'] for r in rs})==500
 assert all(r['status']=='completed' and r['page_after']['resident']==0 and r['application_payload_empty'] and r['gpu_mib_after']==r['cpu_mib_after']==0 for r in rs)
 assert all(r['io_first_decoder']['write_bytes']==0 for r in rs)
 out['cold_audit'][arm]={'completed':len(rs),'all_pre_request_pages_zero':True,'all_application_payload_empty':True,'mean_setup_ms_outside_ttft':statistics.mean(r['setup_ms_outside_ttft'] for r in rs),
 'mean_process_storage_read_bytes_first_decoder':statistics.mean(r['io_first_decoder']['read_bytes'] for r in rs),'mean_major_faults_first_decoder':statistics.mean(r['io_first_decoder']['major_faults'] for r in rs),'io_scope':'process accounting during first decoder, not hardware H2D or NAND traffic'}
 warm=records('v23_full_'+arm,500)
 out['cold_full500'][arm+'_cold_vs_warm_quality']=pair(warm,coldrows[arm])
out['cold_controls']={}
for mode in ('page-hot','ssd-cold'):
 for arm in ('identity','joint'):
  name=f'v23_control_{mode}_{arm}';rs=records(name,8);logs=lines(ROOT/f'results/{name}.cold.jsonl')
  assert len(logs)==8 and all(r['status']=='completed' for r in logs)
  assert all(r['page_after']['resident']==(0 if mode=='ssd-cold' else r['page_after']['pages']) for r in logs)
  out['cold_controls'][mode+'_'+arm]={**basic(rs),'mean_process_storage_read_bytes':statistics.mean(r['io_first_decoder']['read_bytes'] for r in logs),'preserve_policy':True,'cold_log_preconditions_passed':True}
out['calibration']={}
for a in ('joint','independent','query','query_full_validation'):
 x=json.loads((ROOT/f'assets/v23_{a}/trec.pt.report.json').read_text())
 out['calibration'][a]={'best_step':x['best_step'],'report':x['report'],'history':x['history'],'training_args':x['args']}
(ROOT/'results/v23_final_analysis.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'warm':out['full500_warm']['variants'],'warm_timing':out['warm_paired_timing'],'cold':out['cold_full500']['variants'],'cold_timing':out['cold_full500']['paired_timing']},indent=2))
