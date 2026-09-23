import torch,runpy
torch.backends.cuda.matmul.allow_tf32=False
torch.set_float32_matmul_precision('highest')
_svd=torch.linalg.svd
def accurate_svd(x,*a,**kw):
 if x.is_cuda:kw['driver']='gesvd'
 return _svd(x,*a,**kw)
torch.linalg.svd=accurate_svd
runpy.run_path('scripts/compile_prefix_baseline_v27.py',run_name='__main__')
