#!/usr/bin/env python3
import json,csv,statistics,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(path):return [json.loads(x) for x in path.read_text().splitlines()]
configs=['original','split_identity','split_shared','meta_identity','meta_shared','packed_shared','hybrid_identity','hybrid_shared','hybrid_seq3','demand_identity','demand_shared']
variants={};records={};uid_hashes=json.loads((ROOT/'results/v21_query_hash_manifest.json').read_text())
for name in configs:
 blocks=[]
 for block in range(4):
  path=ROOT/f'results/v21_time_b{block}_{name}'
  rs=read(path/'scored_records.jsonl');assert len(rs)==32
  records[(name,block)]={r['uid']:r for r in rs}
  progress=read(ROOT/'logs/v21_timing_jobs.progress.jsonl')
  mem=[r for r in progress if str(path.relative_to(ROOT)) in r['args']]
  assert len(mem)==1
  row={'block':block,'mean_ttft_ms':statistics.mean(r['ttft_ms'] for r in rs),'p95_ttft_ms':float(np.quantile([r['ttft_ms'] for r in rs],.95)),'accuracy_pct':statistics.mean(r['correct'] for r in rs)*100,'physical_payload_bytes':statistics.mean(r['physical_prefetch_kv_bytes'] for r in rs),'host_payload_source_bytes':statistics.mean((r['prefetch_cpu_source_tensor_tokens']+r['prefetch_disk_source_tensor_tokens'])*4*128*2 for r in rs),'cuda_peak_allocated_bytes':mem[0]['cuda_peak_allocated_bytes']}
  extra=path.with_name(path.name+'.extra.jsonl.v21.jsonl')
  if extra.exists():
   es=read(extra)[-32:];assert len(es)==32
   row['selector_calls_prefill']=sorted({r['selector_calls_prefill'] for r in es})
   for k in ('cu_seqlens_cache_hits','cu_seqlens_cache_misses','cu_seqlens_resident_bytes','packed_reader_extra_storage_bytes'):
    if k in es[0]:row[k]=statistics.mean(r[k] for r in es)
  blocks.append(row)
 variants[name]={'mean_ttft_ms':statistics.mean(r['mean_ttft_ms'] for r in blocks),'mean_p95_ttft_ms':statistics.mean(r['p95_ttft_ms'] for r in blocks),'blocks':blocks}
 with (ROOT/f'results/v21_timing_{name}.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['block_id','task','query_hash','ttft_ms']);w.writeheader()
  for block in range(4):
   for uid,r in records[(name,block)].items():w.writerow({'block_id':block,'task':r['task'],'query_hash':uid_hashes[uid],'ttft_ms':r['ttft_ms']})
pairs=[('original','hybrid_seq3'),('hybrid_identity','hybrid_seq3'),('original','hybrid_shared'),('original','hybrid_identity'),('hybrid_identity','hybrid_shared'),('split_identity','split_shared'),('meta_identity','meta_shared'),('meta_shared','packed_shared'),('packed_shared','hybrid_shared'),('split_shared','meta_shared'),('meta_identity','demand_identity'),('packed_shared','demand_shared'),('demand_identity','demand_shared')]
comparisons=[]
for a,b in pairs:
 da=[];bb=[];ss=[]
 for block in range(4):
  x,y=records[(a,block)],records[(b,block)];assert set(x)==set(y)
  av=statistics.mean(r['ttft_ms'] for r in x.values());bv=statistics.mean(r['ttft_ms'] for r in y.values())
  bb.append(av);ss.append(bv);da.append(bv-av)
 rng=np.random.default_rng(19);idx=rng.integers(0,4,(5000,4));boot=np.array(da)[idx].mean(1)
 comparisons.append({'baseline':a,'student':b,'n_blocks':4,'queries_per_block':32,'baseline_mean_ms':statistics.mean(bb),'student_mean_ms':statistics.mean(ss),'relative_latency_change_pct':100*(statistics.mean(ss)/statistics.mean(bb)-1),'delta_per_block_ms':da,'block_bootstrap_delta95_ms':np.quantile(boot,[.025,.975]).tolist()})
out={'scope':'No profiler or logits dumping. Four repeated blocks of the same 32 regression queries, eight warmups per variant, fresh Pcache store. Frozen model reused. Forward/reverse/reverse/forward order. Device 2. Four blocks give limited run-to-run confidence; other users on GPUs0/1 may cause host contention.','variants':variants,'comparisons':comparisons}
(ROOT/'results/v21_timing_summary.json').write_text(json.dumps(out,indent=2))
for r in comparisons:print(r['baseline'],r['student'],round(r['relative_latency_change_pct'],2),r['block_bootstrap_delta95_ms'])
