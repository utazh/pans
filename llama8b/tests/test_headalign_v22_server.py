"""Real Qwen geometry, native decode-tail causality and physical input identity."""
import pytest,torch
from contiguous_fuxian.sharedkv.core import Reader,attention_with_lse
from contiguous_fuxian.sharedkv_v21.ops import pack
from contiguous_fuxian.sharedkv_v21.backend import attention
from contiguous_fuxian.sharedkv_v22.ops import prepare_route,mapped_attention

@pytest.mark.parametrize('t,nt',[(1,42),(41,41),(3,45)])
@pytest.mark.parametrize('p',[[1,0,3,2],[0,3,2,1]])
def test_qwen_cuda_mapped_reader(t,nt,p):
 if not torch.cuda.is_available():pytest.skip('CUDA required')
 torch.manual_seed(22)
 def rnd(*shape):return torch.randn(*shape,device='cuda',dtype=torch.bfloat16)*0.2
 q=rnd(1,t,28,128);pk=rnd(1,1228,4,128);pv=rnd(1,1228,4,128)
 tk=rnd(1,nt,4,128);tv=rnd(1,nt,4,128)
 eye=torch.eye(128,device='cuda',dtype=torch.bfloat16)[None].repeat(4,1,1)
 r=Reader(eye+rnd(4,128,128)*.1,eye+rnd(4,128,128)*.1,rnd(4,128),rnd(4,128))
 seen=[]
 def spy(qi,ki,vi,**kw):
  seen.append((ki,vi));return attention(qi,ki,vi,**kw)
 out=mapped_attention(q,pk,pv,tk,tv,reader=r,route=prepare_route(p,q_heads=28,device='cuda'),packed=pack(r),backend='vllm2',attention_fn=spy)
 assert seen[0][0] is pk and seen[0][1] is pv
 assert seen[1][0] is tk and seen[1][1] is tv
 kk=torch.einsum('bnhd,hde->bnhe',pk[:,:,p].float(),r.A.float())+r.bk.float()
 vv=torch.einsum('bnhd,hde->bnhe',pv[:,:,p].float(),r.B.float())+r.bv.float()
 ref=attention_with_lse(q.float(),torch.cat((kk,tk.float()),1),torch.cat((vv,tv.float()),1),causal=True,backend='reference')[0]
 relative=((out.float()-ref).square().sum()/ref.square().sum()).sqrt()
 assert float(relative)<0.01
 assert torch.allclose(out.float(),ref,atol=.006,rtol=.03)
