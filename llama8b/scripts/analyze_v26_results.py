
from v26_common import *
import csv,statistics,torch,math
from contiguous_fuxian.sharedkv_v26.analysis import records,paired_quality
from contiguous_fuxian.sharedkv.core import Reader
from contiguous_fuxian.sharedkv_v21.ops import pack
def readlines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def mean(xs):return statistics.mean(xs)
def readerbytes(path):
 b=torch.load(path,map_location='cpu',weights_only=True);seen={};keep=[]
 for state in b['readers'].values():
  reader=Reader.from_state_dict(state).to(device='cpu',dtype=torch.bfloat16);packed=pack(reader);keep.extend([reader,packed])
  for obj in [reader,packed]:
   for value in vars(obj).values():
    if isinstance(value,torch.Tensor):seen[value.untyped_storage().data_ptr()]=value.untyped_storage().nbytes()
 return sum(seen.values())
members=json.loads((ROOT/'audit/v26_full_eval_membership.json').read_text())
cells=[('p0',t,b) for t in ['sst2','subj','trec','rte'] for b in [5,10,25,50]]+[('p1',t,25) for t in ['sst2','subj','trec','rte']]
matrix=[];detail={}
for pref,task,budget in cells:
 tag=f'v26_{pref}_{task}_b{budget}'+('_canonical' if task=='subj' else '')
 qr=[records(ROOT/f'results/{tag}_quality_{x}/scored_records.jsonl') for x in ['I','S']]
 q=paired_quality(*qr);assert q['n_unique_queries']==members[pref+'-'+task]['full_count']
 for z in qr:
  assert all(x['accuracy_scoring']=='label_continuation_loglikelihood' for x in z.values())
 clean=set(members[pref+'-'+task]['deduplicated_demo_disjoint_uids'])
 cq=paired_quality(*[{k:v for k,v in rows.items() if k[1] in clean} for rows in qr])
 frozen=json.loads((ROOT/f'audit/{tag}_frozen.json').read_text());rb=readerbytes(frozen['checkpoint'])
 payload=readlines(ROOT/f'results/{tag}_quality_S.extra.jsonl.v25.jsonl')
 assert len(payload)==q['n_unique_queries']+4,(tag,len(payload),q['n_unique_queries'])
 payload=payload[4:] # Exclude the four recorded warm-up requests from resource averages.
 assert all(x['full_selected_source_count']==16 and x['native_policy']['blocks']==4 for x in payload)
 row=dict(prefix=pref,task=task,budget=budget,n=q['n_unique_queries'],identity_accuracy=100*q['baseline_correct']/q['n_unique_queries'],shared_accuracy=100*q['student_correct']/q['n_unique_queries'],delta_pp=q['net_delta_pp'],harmful=q['harmful'],beneficial=q['beneficial'],ni_lower_pp=q['conservative_lower_pp'],exploratory_ni_pass=q['legacy_bound_passes'],source_count=16,native_blocks=4,selected_kv_reduction_pct=100*mean(x['selected_payload_reduction_fraction'] for x in payload),reader_packed_unique_bytes=rb,net_selected_plus_reader_reduction_pct=100*mean(1-(x['total_selected_payload_bytes']+rb)/x['logical_selected_payload_bytes'] for x in payload))
 timings={}
 for condition in ['preserve','warm']:
  ts=[records(ROOT/f'results/{tag}_timing_{condition}_{x}/scored_records.jsonl') for x in ['I','S']]
  assert set(ts[0])==set(ts[1]) and len(ts[0])==128
  vals=[mean(x['ttft_ms'] for x in z.values()) for z in ts]
  row[condition+'_identity_ms']=vals[0];row[condition+'_shared_ms']=vals[1];row[condition+'_delta_pct']=100*(vals[1]/vals[0]-1)
  timings[condition]=dict(n=128,identity_ms=vals[0],shared_ms=vals[1],scope='descriptive single fresh process per arm; no independent repeated-process CI')
  if condition=='preserve':
   for arm in ['I','S']:
    logs=readlines(ROOT/f'results/{tag}_timing_{condition}_{arm}.cold.jsonl')
    assert len(logs)==128 and all(x['status']=='completed' and x['page_after']['resident']==0 and x['application_payload_empty'] for x in logs)
    timings[condition][arm+'_mean_process_read_bytes']=mean(x['io_first_decoder']['read_bytes'] for x in logs)
    timings[condition][arm+'_cold_certified']=len(logs)
 costpath=ROOT/f'audit/{tag}_compute.json'
 if costpath.exists():
  cost=json.loads(costpath.read_text())['wall_seconds_total_including_load']+sum(json.loads((ROOT/f'audit/{tag}_capture_{s}_process.json').read_text())['wall_seconds'] for s in ['fit','val'])
  saved=(row['preserve_identity_ms']-row['preserve_shared_ms'])/1000
  row['reader_build_seconds']=cost;row['cold_payback_requests']=math.ceil(cost/saved) if saved>0 else None
 else:row['reader_build_seconds']=None;row['cold_payback_requests']=None
 planpath=ROOT/f'configs/{tag}_plan.json'
 if not planpath.exists():planpath=ROOT/'configs/v25_explore/g16.json'
 sources=json.loads(planpath.read_text())['sources'];groups={str(src):[i for i,s in enumerate(sources) if s==src and i!=src] for src in sorted(set(sources))};groups={k:v for k,v in groups.items() if v}
 row['active_source_groups']=len(groups);row['participating_layers']=12+len(groups);row['max_source_distance']=max(abs(i-s) for i,s in enumerate(sources))
 detail[tag]=dict(full_quality=q,clean_quality=cq,timing=timings,frozen=frozen,source_map=sources,shared_groups=groups)
 matrix.append(row)
with (ROOT/'results/v26_matrix.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(matrix[0]));w.writeheader();w.writerows(matrix)
with (ROOT/'results/v26_matrix_analysis.json').open('x') as f:json.dump(dict(matrix=matrix,detail=detail),f,indent=2)
# Mechanism quality is paired to the frozen learned/native and identity checkpoints.
baseline=records(ROOT/'results/v26_p0_trec_b25_quality_I/scored_records.jsonl')
learned=records(ROOT/'results/v26_p0_trec_b25_quality_S/scored_records.jsonl')
mechanisms={}
for arm,path in [('learned_native',ROOT/'results/v26_p0_trec_b25_quality_S'),('unit_native',ROOT/'results/v26_mechanism_direct'),('learned_proxy',ROOT/'results/v26_mechanism_proxy')]:
 z=records(path/'scored_records.jsonl');mechanisms[arm]=dict(vs_identity=paired_quality(baseline,z),vs_learned_native=paired_quality(learned,z),diagnostic_only=arm!='learned_native')
with (ROOT/'results/v26_mechanism_analysis.json').open('x') as f:json.dump(mechanisms,f,indent=2)
costs={}
for pref,task,budget in cells:
 tag=f'v26_{pref}_{task}_b{budget}'+('_canonical' if task=='subj' else '');p=ROOT/f'audit/{tag}_compute.json'
 if p.exists():
  cost=json.loads(p.read_text());capture=sum(json.loads((ROOT/f'audit/{tag}_capture_{s}_process.json').read_text())['wall_seconds'] for s in ['fit','val'])
  costs[tag]=dict(teacher_capture_seconds=capture,initial_fit_training_validation_model_loading_seconds=cost['wall_seconds_total_including_load'],total_seconds=capture+cost['wall_seconds_total_including_load'],gpu_peak_allocated=cost['cuda_peak_allocated'],forward_counts=cost['forward_calls_including_validation'])
 else:costs[tag]=dict(status='reused frozen historical v25 checkpoint; prior G26 initialization/search costs not repeated or counted as zero')
with (ROOT/'results/v26_offline_costs.json').open('x') as f:json.dump(costs,f,indent=2)
parser_comparison={}
for arm in ['I','S']:
 legacy=records(ROOT/f'results/v26_p1_subj_b25_quality_{arm}/scored_records.jsonl')
 corrected=records(ROOT/f'results/v26_p1_subj_b25_canonical_quality_{arm}/scored_records.jsonl')
 correction=json.loads((ROOT/'audit/v26_subj_canonical_correction.json').read_text());changed={x['uid'] for x in correction['prefixes']['p1']['repairs']['full_eval']}
 common={k for k in legacy if k[1] not in changed};a={k:legacy[k] for k in common};b={k:corrected[k] for k in common}
 totals=dict(n_unique_queries=len(legacy),baseline_correct=sum(x['correct'] for x in legacy.values()),student_correct=sum(x['correct'] for x in corrected.values()),scope='Descriptive totals: two query texts differ, so no paired interval on this full set.')
 parser_comparison[arm]=dict(comparison=totals,unchanged_query_comparison=paired_quality(a,b),unchanged_token_scores_exact=all(a[k]['label_candidate_token_logprobs']==b[k]['label_candidate_token_logprobs'] for k in common),scope='Parser correction plus refitting 3 corrected calibration/validation records; not a pure inference ablation. Two full-eval records repaired; membership and seed unchanged; retraining numeric variability is not separately isolated.')
with (ROOT/'results/v26_subj_parser_sensitivity.json').open('x') as f:json.dump(parser_comparison,f,indent=2)
import re
run_cost={}
for p in (ROOT/'audit').glob('v26*_process.json'):
 name=p.name
 category=None
 if re.fullmatch(r'v26_p[01]_.+_capture_(fit|val)_process.json',name):category='teacher_capture_including_superseded_SUBJ'
 elif re.fullmatch(r'v26_p[01]_.+_train_process.json',name):category='reader_training_including_superseded_SUBJ'
 elif re.fullmatch(r'v26_p[01]_.+_quality_[IS]_process.json',name):category='full_quality_including_legacy_SUBJ'
 elif re.fullmatch(r'v26_p[01]_.+_timing_(preserve|warm)_[IS]_process.json',name):category='matrix_timing'
 elif name=='v26_primary_study_process.json':category='primary_study_serial_total'
 elif name in ['v26_mechanism_direct_process.json','v26_mechanism_proxy_process.json']:category='mechanism_evaluation'
 elif name.startswith('v26_profile_'):category='profilers_and_exports'
 if category:
  value=json.loads(p.read_text());assert value['returncode']==0
  item=run_cost.setdefault(category,dict(runs=0,process_wall_seconds=0.0));item['runs']+=1;item['process_wall_seconds']+=value['wall_seconds']
with (ROOT/'results/v26_run_cost_summary.json').open('x') as f:json.dump(dict(categories=run_cost,scope='Wall-clock process time including model load/CPU wait; not GPU kernel time. Concurrent processes overlap, so sums are not total elapsed time. Includes discarded legacy SUBJ calibration. Excludes manual preparation, prior-version R&D and setup/audit runs not represented here. Reader deployment table separately excludes experimental repeats.'),f,indent=2)
print('MATRIX ANALYZED',len(matrix))
