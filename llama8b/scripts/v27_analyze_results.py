
from v26_common import *
from contiguous_fuxian.sharedkv_v27.quality import load_records,paired_quality
import csv,statistics,torch
names=['v27_online_I','v27_online_native','v27_online_proxy','v27_online_r0','v27_kvsharer_g24_quality','v27_kvsharer_g20_quality','v27_kvsharer_g16_quality','v27_clf_r256_384_quality','v27_clf_r384_576_quality','v27_clf_bf16_256_384_quality','v27_clf_bf16_384_576_quality']
baseline=load_records(ROOT/f'results/{names[0]}/scored_records.jsonl');out=[]
for name in names:
 path=ROOT/f'results/{name}/scored_records.jsonl'
 if not path.exists():continue
 rs=load_records(path);assert len(rs)==500
 pair=paired_quality(baseline,rs)
 pair.update(name=name,correct=sum(x['correct'] for x in rs.values()),first_token_label_agreement=sum(x['label_first_token_prediction']==baseline[k]['label_first_token_prediction'] for k,x in rs.items()),selection_hash_exact_agreement=sum(x['layer_token_selection_sha256']==baseline[k]['layer_token_selection_sha256'] for k,x in rs.items()),external_latency_valid=False if ('kvsharer' in name or 'clf' in name) else None)
 out.append(pair)
report={'scope':'Qwen2.5-7B TREC p0/25%; previously observed 500 queries; exploratory prefix adaptation, not external official speed reproduction','quality':out,'missing':[n for n in names if not (ROOT/f'results/{n}/scored_records.jsonl').exists()]}
(ROOT/'results/v27_quality_analysis.json').write_text(json.dumps(report,indent=2))
with (ROOT/'results/v27_quality.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['name','n','correct','baseline_accuracy','student_accuracy','delta_pp','harmful','beneficial','conservative_lower_pp','certified']);w.writeheader()
 for x in out:w.writerow({k:x[k] for k in w.fieldnames})
print(json.dumps([{'name':x['name'],'correct':x['correct'],'delta_pp':x['delta_pp']} for x in out]))
if (ROOT/'audit/v27_cold_done.json').exists():
 matrix={};certs=0;uids=set()
 for block in range(3):
  base=None
  for arm in ['I','native','r0']:
   name=f'v27_cold_b{block}_{arm}';rs=load_records(ROOT/f'results/{name}/scored_records.jsonl')
   assert len(rs)==32
   if base is None:base=set(rs);assert not uids&base;uids|=base
   else:assert set(rs)==base
   cc=[json.loads(l) for l in (ROOT/f'results/{name}.cold.jsonl').read_text().splitlines()]
   measured=[x for x in cc if x.get('measured')]
   assert len(measured)==32
   assert all(x['status']=='completed' and x['application_payload_empty'] and x['page_after']['resident']==0 for x in measured)
   certs+=len(measured)
   matrix[(block,arm)]=rs
 result={'certified_cold_requests':certs,'unique_queries':len(uids),'blocks':3,'cache_condition':'ssd-cold/preserve','admission':'reuse-prefetch','scope':'Latin-square position balance, three disjoint blocks; no stable/general speed certification','arms':{}}
 for arm in ['I','native','r0']:
  vals=[x['ttft_ms'] for block in range(3) for x in matrix[(block,arm)].values()]
  means=[statistics.mean(x['ttft_ms'] for x in matrix[(block,arm)].values()) for block in range(3)]
  basevals=[x['ttft_ms'] for block in range(3) for x in matrix[(block,'I')].values()]
  result['arms'][arm]=dict(mean_ms=statistics.mean(vals),median_ms=statistics.median(vals),block_means_ms=means,delta_pct=100*(statistics.mean(vals)/statistics.mean(basevals)-1),n=len(vals))
 (ROOT/'results/v27_cold_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
