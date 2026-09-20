import pytest,torch
from contiguous_fuxian.sharedkv.core import Reader
from contiguous_fuxian.sharedkv_v21.ops import pack,dispatch
from contiguous_fuxian.sharedkv_v21.backend import LengthPool,_ACTIVE_POOL
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('t,tail',[(1,33),(17,17),(3,29)])
def test_real_gqa_hybrid_and_packed_vllm(t,tail):
 pytest.importorskip('vllm.vllm_flash_attn')
 torch.manual_seed(21)
 def rand(n,h):return torch.randn(1,n,h,128,device='cuda',dtype=torch.bfloat16)*.2
 q=rand(t,28);pk=rand(113,4);pv=rand(113,4);tk=rand(tail,4);tv=rand(tail,4)
 eye=torch.eye(128,device='cuda',dtype=torch.bfloat16).repeat(4,1,1)
 bias=torch.randn(4,128,device='cuda',dtype=torch.bfloat16)*.02
 r=Reader(eye+torch.randn_like(eye)*.01,eye+torch.randn_like(eye)*.01,bias,bias)
 pool=LengthPool();token=_ACTIVE_POOL.set(pool)
 try:
  ref=dispatch(q,pk,pv,tk,tv,backend='reference')
  split=dispatch(q,pk,pv,tk,tv,backend='vllm2')
  hybrid=dispatch(q,pk,pv,tk,tv,backend='vllm2',layout='hybrid')
  old=dispatch(q,pk,pv,tk,tv,reader=r,backend='vllm2')
  new=dispatch(q,pk,pv,tk,tv,reader=r,packed=pack(r),backend='vllm2')
  torch.testing.assert_close(split,ref,atol=.002,rtol=.03)
  torch.testing.assert_close(hybrid,ref,atol=.002,rtol=.03)
  torch.testing.assert_close(new,old,atol=.002,rtol=.03)
  assert pool.hits>0 and pool.misses<12
 finally:_ACTIVE_POOL.reset(token)
