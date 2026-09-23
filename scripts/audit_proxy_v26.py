
from pathlib import Path
import argparse,runpy,sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from contiguous_fuxian.sharedkv_v26 import control
p=argparse.ArgumentParser();p.add_argument('--rounding-audit',required=True);p.add_argument('rest',nargs=argparse.REMAINDER);a=p.parse_args()
args=a.rest[1:] if a.rest[:1]==['--'] else a.rest
old=control.source_proxy;seen=set();rows=[]
def proxy(k,v,cut,reader,**kw):
 result=old(k,v,cut,reader,**kw)
 if id(reader) not in seen:
  seen.add(id(reader));item={'patch_tokens':k.shape[1]-cut,'whole_prefix_reconstructed':False}
  for name,src,mat,bias,actual in [('k',k,reader.A,reader.bk,result[0]),('v',v,reader.B,reader.bv,result[1])]:
   reference=torch.einsum('bnhd,hde->bnhe',src[:,cut:].float(),mat.float())+bias.float()[None,None]
   err=actual.float()-reference
   item[name]=dict(max_abs_error=float(err.abs().max()),relative_l2=float(err.norm()/reference.norm().clamp_min(1e-12)),reference='FP32 contraction and addition on the exact same BF16 source/readers; proxy remains BF16')
  rows.append(item)
 return result
control.source_proxy=proxy
try:
 sys.argv=[str(ROOT/'scripts/run_evidence_v26.py'),*args];runpy.run_path(sys.argv[0],run_name='__main__')
finally:
 control.source_proxy=old
 with Path(a.rounding_audit).open('x') as f:json.dump(rows,f,indent=2)
