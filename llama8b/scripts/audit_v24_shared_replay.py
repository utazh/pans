"""Diagnostic shared scorer/replay parity on the same online selected IDs."""
import argparse,json,runpy,sys
from pathlib import Path
import torch
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'src'))
p=argparse.ArgumentParser();p.add_argument('--audit-output',required=True);p.add_argument('rest',nargs=argparse.REMAINDER);a=p.parse_args()
argv=a.rest[1:] if a.rest[0]=='--' else a.rest
out=Path(a.audit_output);assert not out.exists()
from contiguous_fuxian.sharedkv_v21.runtime import Runtime
from contiguous_fuxian.sharedkv.io import load_safe,read_prefix,query_hash
from contiguous_fuxian.sharedkv.core import Reader
from contiguous_fuxian.sharedkv_v23.block import TrainableReader
from contiguous_fuxian.sharedkv_v24.replay import selected_bank,continuation_tokens
old=Runtime.install;state={}
def install(runtime,runner):
 old(runtime,runner);previous=runner.greedy_flexgen_completion
 def completion(*args,**kw):
  tokenlp=kw.get('label_token_logprobs_out')
  if tokenlp is None:tokenlp={};kw['label_token_logprobs_out']=tokenlp
  result=previous(*args,**kw)
  model,loader=kw['model'],kw['loader'];device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
  if not state:
   bundle=load_safe(Path(argv[argv.index('--adapters')+1])/'trec.pt')
   state['sources']=bundle['sources']
   state['readers']=torch.nn.ModuleDict({k:TrainableReader(Reader.from_state_dict(v)) for k,v in bundle['readers'].items()}).to(device)
   state['raw']={lid:tuple(read_prefix(r/'runtime_data/prefix_store','trec',lid,kind)[0].to(device=device,dtype=dtype) for kind in ['key','value']) for lid in range(28)}
   state['geometry']=bundle['geometry']
  row={'query_token_ids':kw['query_token_ids'],'label_ids':kw['label_continuations'],'prefix_tokens':loader.prefix_tokens,'teacher_backend':runtime.backend,'teacher_layout':runtime.layout,'selection_by_layer':[list(map(int,x)) for x in loader._selected_by_layer],'period':kw['impress_selection_period_size'],'geometry':state['geometry']}
  with torch.no_grad():
   bank=selected_bank(state['raw'],row,state['sources'],device,dtype)
   got={label:continuation_tokens(model,row,label,state['sources'],bank,state['readers'],checkpoint_layers=False).cpu() for label in row['label_ids']}
  errors={label:float((got[label]-torch.tensor(tokenlp[label])).abs().max()) for label in got}
  means={label:abs(float(got[label].mean())-sum(tokenlp[label])/len(tokenlp[label])) for label in got}
  record={'query_hash':query_hash(kw['query_token_ids']),'max_token_error':max(errors.values()),'max_mean_score_error':max(means.values()),'by_label_token_error':errors,'same_actual_online_selection':True,'formal_timing_eligible':False}
  with out.open('a') as f:f.write(json.dumps(record)+'\n')
  assert record['max_token_error']<=.10 and record['max_mean_score_error']<=.02,record
  return result
 runner.greedy_flexgen_completion=completion
Runtime.install=install
try:
 sys.argv=[str(r/'scripts/run_sharedkv_v23.py'),*argv];runpy.run_path(sys.argv[0],run_name='__main__')
finally:Runtime.install=old
