#!/usr/bin/env python3
"""Audit only this experiment's own generated captures (trusted torch files)."""
import argparse,json,csv,statistics,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from contiguous_fuxian.sharedkv_v21.audit import paired_quality,mean_label_scores
p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(4)
def rows(path):return [json.loads(x) for x in (path/'scored_records.jsonl').read_text().splitlines()]
def captures(path):
 return {p.stem:torch.load(p,map_location='cpu',weights_only=False) for p in path.glob('*.pt')}
base=ROOT/'results'/a.baseline
b=rows(base);bc=captures(base.with_name(base.name+'_logits')); result={}
for path in sorted((ROOT/'results').glob(a.prefix+'*')):
 if not path.is_dir() or not (path/'summary.json').exists():continue
 s=rows(path)
 if {r['uid'] for r in s}!={r['uid'] for r in b}:continue
 data=paired_quality([dict(r,query_hash=r['uid']) for r in b],[dict(r,query_hash=r['uid']) for r in s])
 data['accuracy_pct']=100*statistics.mean(int(r['correct']) for r in s)
 data['diagnostic_ttft_ms']=statistics.mean(r['ttft_ms'] for r in s)
 data['physical_payload_bytes']=statistics.mean(r['physical_prefetch_kv_bytes'] for r in s)
 data['host_payload_source_bytes']=statistics.mean((r['prefetch_cpu_source_tensor_tokens']+r['prefetch_disk_source_tensor_tokens'])*4*128*2 for r in s)
 bx={r['uid']:r for r in b}
 data['harmful_uids']=[r['uid'] for r in s if bx[r['uid']]['correct'] and not r['correct']]
 data['beneficial_uids']=[r['uid'] for r in s if not bx[r['uid']]['correct'] and r['correct']]
 data['whole_plan_exact_fraction']=statistics.mean(r['layer_token_selection_sha256']==bx[r['uid']]['layer_token_selection_sha256'] for r in s)
 sc=captures(path.with_name(path.name+'_logits'))
 if sc:
  assert set(bc)==set(sc) and len(sc)==len(s),(path,len(sc),len(s))
  kl=[]; jac=[[] for _ in range(28)]; labels=[]
  for h in sorted(bc):
   x,y=bc[h],sc[h];xx=x['logits'].float().log_softmax(-1);yy=y['logits'].float().log_softmax(-1)
   kl.append(float((xx.exp()*(xx-yy)).sum()))
   for lid,(aa,bb) in enumerate(zip(x['selected_by_layer'],y['selected_by_layer'])):
    aa,bb=set(aa),set(bb);jac[lid].append(len(aa&bb)/len(aa|bb))
   ax,ay=x.get('label_token_logprobs'),y.get('label_token_logprobs')
   if ax is not None and ay is not None:
    assert set(ax)==set(ay)
    keys=sorted(ax);xx=torch.tensor([mean_label_scores(ax)[k] for k in keys]);yy=torch.tensor([mean_label_scores(ay)[k] for k in keys])
    vals=xx.sort(descending=True).values
    labels.append({'query_hash':h,'baseline_margin':float(vals[0]-vals[1]),'student_margin':float(yy.topk(2).values.diff().abs()[0]),'argmax_same':bool(xx.argmax()==yy.argmax()),'max_abs_delta':float((xx-yy).abs().max()),'kl':float((xx.softmax(-1)*(xx.log_softmax(-1)-yy.log_softmax(-1))).sum())})
  data['first_token_mean_kl']=statistics.mean(kl);data['first_token_p95_kl']=float(np.quantile(kl,.95))
  data['token_jaccard_by_layer']=[{'mean':statistics.mean(v),'min':min(v)} for v in jac]
  data['label_score_reduction']='mean per token, matching actual runner decision'
  data['label_mean_kl']=statistics.mean(r['kl'] for r in labels) if labels else None
  data['label_argmax_agreement']=statistics.mean(r['argmax_same'] for r in labels) if labels else None
  data['label_diagnostics']=labels
  data['query_tokens']={'mean':statistics.mean(r['query_tokens'] for r in sc.values()),'min':min(r['query_tokens'] for r in sc.values()),'max':max(r['query_tokens'] for r in sc.values())}
 stats=path.with_name(path.name+'.extra.jsonl')
 if stats.exists():
  ss=[json.loads(x) for x in stats.read_text().splitlines()][-len(s):]
  data['physical_alias_all']=all(all(r['prefix_storage_ptrs_by_layer'][i]==r['prefix_storage_ptrs_by_layer'][source] for i,source in enumerate(r['sources'])) for r in ss)
  data['private_tail_bounded_all']=all(max(r['private_tail_lengths'].values())<=r['query_tokens']+r['requested_max_tokens']+4 for r in ss)
  for k in ('physical_resolve_calls','request_prefix_buffer_bytes','resident_reader_bytes','cuda_peak_allocated_bytes'):
   data[k]=max(r[k] for r in ss)
 result[path.name]=data
 print(path.name,round(data['accuracy_pct'],3),round(data.get('first_token_mean_kl',0),5),'harm',data['baseline_right_student_wrong'],'gain',data['baseline_wrong_student_right'],flush=True)
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'baseline':a.baseline,'scope':'quality only; captures invalidate formal latency claims','variants':result},indent=2))
with out.with_suffix('.csv').open('w') as f:
 keys=['variant','n','accuracy_pct','delta_pp','baseline_right_student_wrong','baseline_wrong_student_right','first_token_mean_kl','label_mean_kl','whole_plan_exact_fraction','physical_payload_bytes','host_payload_source_bytes']
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
 for name,r in result.items():w.writerow({k:name if k=='variant' else r.get(k) for k in keys})
