import torch
from contiguous_fuxian.sharedkv_v28.common import save_tensor,load_tensor
def test_lossless_direct_reference(tmp_path):
 value={'method':'direct-map-prefix-quality','native':{'0':[torch.randn(80,4,8).bfloat16().float(),torch.zeros(80,4,8)]},'sources':[0]}
 path=tmp_path/'reference.pt';save_tensor(path,value);restored=load_tensor(path)
 assert restored['sources']==value['sources']
 for a,b in zip(value['native']['0'],restored['native']['0']):
  assert a.dtype==b.dtype and torch.equal(a,b)
