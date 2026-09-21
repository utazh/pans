
import pytest,torch
from contiguous_fuxian.sharedkv_v24.faithful import attention,probabilities
@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('hk',[1,2])
def test_attention_lse_vjp_gradcheck(causal,hk):
 torch.manual_seed(6)
 q=torch.randn(1,2,4,4,dtype=torch.float64,requires_grad=True)
 k=torch.randn(1,3,hk,4,dtype=torch.float64,requires_grad=True)
 v=torch.randn_like(k,requires_grad=True)
 assert torch.autograd.gradcheck(lambda q,k,v:attention(q,k,v,causal),(q,k,v),atol=1e-5,rtol=1e-4)
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
@pytest.mark.parametrize('causal',[False,True])
def test_cuda_forward_exact_and_vjp(causal):
 from contiguous_fuxian.sharedkv_v21.backend import attention as online
 torch.manual_seed(6)
 q=torch.randn(1,3,8,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 k=torch.randn(1,7,2,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 v=torch.randn_like(k,requires_grad=True)
 out,lse=attention(q,k,v,causal)
 ref,rz=online(q.detach(),k.detach(),v.detach(),causal=causal,backend='vllm2')
 assert torch.equal(out,ref) and torch.equal(lse,rz)
 go=torch.randn_like(out);gz=torch.randn_like(lse)
 got=torch.autograd.grad((out,lse),(q,k,v),(go,gz))
 qq,kk,vv=[x.detach().float().requires_grad_() for x in (q,k,v)]
 _,_,p,z=probabilities(qq,kk,causal)
 expected=(p@vv.permute(0,2,1,3).unsqueeze(2)).permute(0,3,1,2,4).reshape_as(qq)
 want=torch.autograd.grad((expected,z),(qq,kk,vv),(go.float(),gz))
 for a,b in zip(got,want):torch.testing.assert_close(a.float(),b.to(a.dtype).float(),atol=.002,rtol=.01)
