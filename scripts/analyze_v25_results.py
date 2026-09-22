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

def compare(a,b,alpha=.05):
 aa={v['uid']:v for v in a};bb={v['uid']:v for v in b};d=paired_report(aa,bb,alpha=alpha)
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
 return {'paired_blocks':len(blocks),'distinct_query_ids':len({v['uid'] for a,b in blocks for v in a}),'baseline_mean_ms':float(bm.mean()),'shared_mean_ms':float(sm.mean()),'relative_change_pct':float(100*(sm.mean()/bm.mean()-1)),'block_bootstrap_delta95_ms':np.quantile(delta,[.025,.975]).tolist(),'block_bootstrap_relative95_pct':np.quantile(rel,[.025,.975]).tolist(),'per_block':[{'identity_ms':float(a),'shared_ms':float(b),'delta_ms':float(b-a)} for a,b in zip(bm,sm)],'limitation':f'{len(blocks)} blocks repeating {len(blocks[0][0])} queries; block bootstrap is exploratory, not independent repeat-query samples.'}

protocol=json.loads((r/'audit/v25_preregistered_plan.json').read_text())
freeze=json.loads((r/'audit/v25_frozen_selection.json').read_text());G=freeze['chosen_G']
out={'protocol':protocol,'frozen_selection':freeze,'quality':{},'quality_scope':'Final1536 is a newly held-out training-pool sample. Historical500 and reused exploration768 are not fresh certification data. Concurrent quality-run TTFT is excluded.'}
for split,n,prefix,alpha in [('dev',768,'v25_dev_',.05),('final',1536,'v25_final_',.025),('test500',500,'v25_test500_',.05)]:
 base=rows(prefix+'identity',n);variants={}
 if split=='final':
  arms=['identity']+[f'g{G}_r{R}_s{s}' for R in [0,4] for s in [24,25,26]]
 else:
  arms=['identity']+[f'g{g}_r{R}_s24' for g in [26,24,22,20,16,14] for R in [0,4]]
  if split=='test500':arms += [f'g{G}_r{R}_s{s}' for R in [0,4] for s in [25,26]]
 for arm in arms:
  name=prefix+arm;z=rows(name,n)
  variants[arm]={'basic':basic(z),'paired_vs_identity':compare(base,z,alpha)}
 out['quality'][split]=variants
out['certification']={}
for R in [0,4]:
 reports=[out['quality']['final'][f'g{G}_r{R}_s{s}']['paired_vs_identity'] for s in [24,25,26]]
 out['certification'][f'r{R}']={'all_three_fixed_seeds_pass':all(x['conservative_bound_passes'] for x in reports),'alpha_per_seed':.025,'margin_pp':1,'scope':'Intersection across three fixed seeds, Bonferroni across two preselected R routes; assumptions of paired binomial bound remain. No automatic deployment.'}
out['final_R4_vs_R0']={}
for s in [24,25,26]:out['final_R4_vs_R0'][str(s)]=compare(rows(f'v25_final_g{G}_r0_s{s}',1536),rows(f'v25_final_g{G}_r4_s{s}',1536))
out['curve_preserve']={}
for g in [26,24,22,20,16,14]:
 for R in [0,4]:
  b=[(rows(f'v25_time_preserve_b{i}_identity',16),rows(f'v25_time_preserve_b{i}_g{g}_r{R}',16)) for i in [0,1]]
  out['curve_preserve'][f'g{g}_r{R}']=timing(b)
out['selected_timing']={}
for policy,n in [('preserve',16),('native',16),('warm',32)]:
 out['selected_timing'][policy]={}
 for R in [0,4]:
  b=[(rows(f'v25_time_{policy}_b{i}_identity',n),rows(f'v25_time_{policy}_b{i}_g{G}_r{R}',n)) for i in range(4)]
  out['selected_timing'][policy][f'r{R}']=timing(b)
out['timing_scope']='Every arm uses a fresh process, same eight warmups. Preserve curve: two whole-grid blocks; selected preserve adds two later selected-only blocks. Selected native/warm have four blocks. Small repeated-query block intervals are exploratory, not hundreds of independent samples.'
out['cold_certification']={}
manifest=json.loads((r/'audit/v25_formal_timing_manifest.json').read_text())
for job in manifest:
 if job['policy']=='warm':continue
 z=lines(r/f"results/{job['name']}.cold.jsonl")
 assert len(z)==job['n']
 assert all(x['status']=='completed' and x['page_after']['resident']==0 and x['application_payload_empty'] and x['gpu_mib_after']==x['cpu_mib_after']==0 and x['io_first_decoder']['write_bytes']==0 for x in z)
 out['cold_certification'][job['name']]={'n':len(z),'all_certified':True,'setup_ms_excluded':statistics.mean(x['setup_ms_outside_ttft'] for x in z),'process_read_bytes_first_decoder':statistics.mean(x['io_first_decoder']['read_bytes'] for x in z)}
out['resource']={}
keys=['logical_selected_payload_bytes','shared_base_payload_bytes','native_repair_payload_bytes','total_selected_payload_bytes','selected_payload_reduction_fraction','full_selected_source_count','partial_native_layer_count','native_repair_resolve_calls','full_source_resolve_calls','total_resolve_calls','native_repair_buffer_bytes','full_source_buffer_bytes']
for label in ['identity']+[f'g{g}_r{R}' for g in [26,24,22,20,16,14] for R in [0,4]]:
 name=f'v25_time_preserve_b0_{label}'
 z=lines(r/f'results/{name}.extra.jsonl.v25.jsonl')[8:];assert len(z)==16
 d={k:statistics.mean(v[k] for v in z) for k in keys}
 base=lines(r/f'results/{name}.extra.jsonl')[8:];packed=lines(r/f'results/{name}.extra.jsonl.v21.jsonl')[8:]
 d['resident_reader_bytes']=statistics.mean(v['resident_reader_bytes'] for v in base)
 d['packed_reader_extra_storage_bytes']=statistics.mean(v['packed_reader_extra_storage_bytes'] for v in packed)
 d['selected_plus_readers_net_reduction_pct']=100*(1-(d['total_selected_payload_bytes']+d['resident_reader_bytes']+d['packed_reader_extra_storage_bytes'])/d['logical_selected_payload_bytes'])
 out['resource'][label]=d
out['resource_scope']='Includes full owner selected buffers plus small consumer native buffers. Selected-buffer net additionally counts resident/packed readers; excludes model weights, selector/index, allocator reserve, temporary concat workspace and private KV. No persistent disk pruning.'
out['training']={}
for p in sorted((r/'audit').glob('v25_g*_compute.json')):
 name=p.name.removesuffix('_compute.json');x=json.loads(p.read_text());assert x['status']=='completed'
 d=json.loads((r/f'assets/{name}/trec.pt.report.json').read_text())
 out['training'][name]={**x,'best_step':d['best_step'],'validation':{k:v for k,v in d['final'].items() if k!='query_diagnostics'},'native_parity':{k:v for k,v in d['parity'].items() if k in ['max_abs_mean_score_error','max_abs_token_logprob_error']}}
out['amortization_training_only']={}
parent=json.loads((r/'audit/v24_score_g26_s24_compute.json').read_text())['wall_seconds_total_including_load']
for R in [0,4]:
 t=out['selected_timing']['preserve'][f'r{R}'];saving=(t['baseline_mean_ms']-t['shared_mean_ms'])/1000
 train=out['training'][f'v25_g{G}_r{R}_s24']['wall_seconds_total_including_load']
 out['amortization_training_only'][f'r{R}']={'v25_training_seconds':train,'inherited_parent_training_seconds':parent,'measured_mean_saving_seconds':saving,'queries_v25_training_only':train/saving if saving>0 else None,'queries_with_parent_training_only':(train+parent)/saving if saving>0 else None,'scope':'Illustrative training-only amortization using observed mean TTFT. Excludes teacher collection, topology/grid search and other preparation; not a total-cost break-even guarantee.'}
out['restore_diagnostic']={}
for policy in ['native','preserve']:
 for arm in ['identity','r0','shared']:
  z=lines(r/f'audit/v25_restore_{policy}_{arm}.jsonl')[4:];assert len(z)==8
  out['restore_diagnostic'][policy+'_'+arm]={'n':8,'prefetch':{k:statistics.mean(v['prefetch'][k] for v in z) for k in z[0]['prefetch']},'promotion':{k:statistics.mean(v['promotion'][k] for v in z) for k in z[0]['promotion']},'mean_files_seen_in_both_paths':statistics.mean(v['files_seen_in_both_paths'] for v in z),'formal_timing_eligible':False}
out['nsight']=json.loads((r/'results/v25_nsight_summary.json').read_text())
out['invariants']=json.loads((r/'audit/v25_final_invariants.json').read_text())
for policy in ['native','warm']:
 for arm in ['identity','shared']:out[f'viz_{policy}_{arm}']=json.loads((r/f'results/v25_viz_{policy}_{arm}_summary.json').read_text())
out['test500_strict495']={}
for arm in out['quality']['test500']:
 z=[v for v in rows('v25_test500_'+arm,500) if v['uid'] not in {'trec-0','trec-32','trec-33','trec-34','trec-35'}];assert len(z)==495
 out['test500_strict495'][arm]=basic(z)
(r/'results/v25_final_analysis.json').write_text(json.dumps(out,indent=2))
with (r/'results/v25_quality_table.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['split','variant','n','correct','accuracy_pct','harmful','beneficial','delta_pp','lower_pp'])
 for split,arms in out['quality'].items():
  for arm,x in arms.items():
   b,p=x['basic'],x['paired_vs_identity'];w.writerow([split,arm,b['n'],b['correct'],b['accuracy_pct'],p['harmful'],p['beneficial'],p['net_delta_pp'],p['conservative_one_sided_lower_pp']])
print(json.dumps({'selected_G':G,'certification':out['certification'],'selected_timing':out['selected_timing'],'cold_requests':sum(x['n'] for x in out['cold_certification'].values())},indent=2))
