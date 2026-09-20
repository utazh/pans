#!/usr/bin/env python3
import sys,json,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from contiguous_fuxian.sharedkv_v21.traces import load_traces,paired
from contiguous_fuxian.sharedkv_v21.audit import audit_traces,audit_pair_geometry,operator_audit
from contiguous_fuxian.sharedkv.io import load_safe
from contiguous_fuxian.sharedkv.core import Reader
torch.set_num_threads(4)
t=load_traces(ROOT/'runtime_data/v21_teacher_val','trec')
for source,target,student,adapter in [(3,2,'v21_teacher_val','v21_refit2_s200'),(18,19,'v21_student2_val','v21_seq2'),(26,27,'v21_student19_val','v21_seq3')]:
 s=t if student=='v21_teacher_val' else load_traces(ROOT/'runtime_data'/student,'trec')
 r=Reader.from_state_dict(load_safe(ROOT/'assets'/adapter/'trec.pt')['readers'][str(target)])
 result={'layer_drift':audit_traces(t,s),'logit_span':audit_pair_geometry(t,store_root=ROOT/'runtime_data/prefix_store',task='trec',source=source,target=target),'operator':operator_audit(paired(t,s,target),store_root=ROOT/'runtime_data/prefix_store',task='trec',source=source,target=target,reader=r)}
 (ROOT/f'results/v21_operator_{target}.json').write_text(json.dumps(result,indent=2))
 print(target,{k:statistics.mean(x[k] for x in result['operator']['rows']) for k in result['operator']['rows'][0] if k!='query_hash'},flush=True)
