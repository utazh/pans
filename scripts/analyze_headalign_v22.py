import json,statistics,sys,hashlib
from pathlib import Path
import torch
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'src'))
from contiguous_fuxian.sharedkv_v21.audit import paired_quality
def read(p):return [json.loads(x) for x in p.read_text().splitlines()]
q=json.loads((r/'results/v22_quality_compare.json').read_text())
pairs={}
for t in (19,27):
 for step in (0,200):
  a=f'v22_reg_diag{t}_s{step}';b=f'v22_reg_map{t}_s{step}'
  ar=read(r/'results'/a/'scored_records.jsonl');br=read(r/'results'/b/'scored_records.jsonl')
  d=paired_quality([dict(x,query_hash=x['uid']) for x in ar],[dict(x,query_hash=x['uid']) for x in br])
  d.update(baseline=a,student=b,baseline_accuracy=100*statistics.mean(x['correct'] for x in ar),student_accuracy=100*statistics.mean(x['correct'] for x in br))
  pairs[f'{t}_s{step}']=d
local={}
for p in sorted((r/'assets').glob('v22_*/trec.report.json')):
 d=json.loads(p.read_text());local[p.parent.name]=d
resources={}
base=q['variants']['v22_reg_identity']['request_prefix_buffer_bytes']
for name,d in q['variants'].items():
 extra=read(r/'results'/f'{name}.extra.jsonl.v21.jsonl')
 route=read(r/'results'/f'{name}.extra.jsonl.v22.jsonl')
 reader=d['resident_reader_bytes'];packed=extra[-1]['packed_reader_extra_storage_bytes'];indices=route[-1]['resident_head_route_bytes']
 total=d['request_prefix_buffer_bytes']+reader+packed+indices
 resources[name]={'prefix_buffer_bytes':d['request_prefix_buffer_bytes'],'reader_bytes':reader,'packed_extra_bytes':packed,'routing_bytes':indices,'prefix_plus_readers_bytes':total,'net_saving_bytes':base-total,'net_saving_pct':100*(base-total)/base,'payload_saving_pct':100*(base-d['request_prefix_buffer_bytes'])/base,'not_total_model_memory':True}
fixed={}
for name in ('diag19_s200','map19_s200'):
 a=read(r/'results/v22_fixed_identity/scored_records.jsonl');b=read(r/f'results/v22_fixed_{name}/scored_records.jsonl')
 assert {x['uid']:x['layer_token_selection_sha256'] for x in a}=={x['uid']:x['layer_token_selection_sha256'] for x in b}
fixed['selector_plan_equal_to_identity']=True
decode={}
for name in ('identity','map19_s200','map27_s200'):
 rows=read(r/f'results/v22_decode_{name}.extra.jsonl')[-8:]
 diffs=[v-x['query_tokens'] for x in rows for v in x['private_tail_lengths'].values()]
 decode[name]={'requests':len(rows),'requested_max_tokens':sorted({x['requested_max_tokens'] for x in rows}),'private_extra_tokens_range':[min(diffs),max(diffs)],'private_prefix_excluded':all(max(x['private_tail_lengths'].values())<x['prefix_tokens'] for x in rows) if 'prefix_tokens' in rows[0] else 'see invariant check'}
out={'pairs':pairs,'local':local,'resources':resources,'fixed':fixed,'decode':decode,'data_status':'All 175 end-to-end requests previously observed in v21; regression evidence, not unseen final test.'}
(r/'results/v22_analysis.json').write_text(json.dumps(out,indent=2))
for k,v in pairs.items():print(k,v)
print('resources',resources['v22_reg_map19_s200'])
