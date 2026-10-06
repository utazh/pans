#!/bin/bash
# GPU sanity on one idle GPU: vLLM flash-attn (varlen + LSE) vs reference, then the repo test-suite.
B=/home/shensicheng/qyd/pans; G=${1:-6}
export CUDA_VISIBLE_DEVICES=$G TMPDIR=$B/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $B/src
$B/venv/bin/python - <<'EOF'
import torch, math
from vllm.vllm_flash_attn import flash_attn_varlen_func
print('device', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
torch.manual_seed(0)
for (tq, tk, hq, hk) in [(17, 1200, 28, 4), (5, 4940, 40, 8)]:
    q = torch.randn(tq, hq, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(tk, hk, 128, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(tk, hk, 128, device='cuda', dtype=torch.bfloat16)
    cu_q = torch.tensor([0, tq], device='cuda', dtype=torch.int32); cu_k = torch.tensor([0, tk], device='cuda', dtype=torch.int32)
    out, lse = flash_attn_varlen_func(q, k, v, max_seqlen_q=tq, cu_seqlens_q=cu_q, max_seqlen_k=tk, cu_seqlens_k=cu_k,
                                      softmax_scale=1 / math.sqrt(128), causal=False, return_softmax_lse=True)
    g = hq // hk
    kk = k.float().repeat_interleave(g, 1); vv = v.float().repeat_interleave(g, 1)
    s = torch.einsum('qhd,khd->hqk', q.float(), kk) / math.sqrt(128)
    ref = torch.einsum('hqk,khd->qhd', s.softmax(-1), vv); rlse = torch.logsumexp(s, -1)
    print(f'q{tq} k{tk} h{hq}/{hk}: out maxdiff {(out.float() - ref).abs().max().item():.3e}  lse maxdiff {(lse.float() - rlse).abs().max().item():.3e}')
print('SANITY_OK')
EOF
$B/venv/bin/python -m pytest -q -x 2>&1 | tail -15
echo SANITY_DONE
