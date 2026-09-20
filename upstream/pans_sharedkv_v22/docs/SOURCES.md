# 阅读与参考代码

以下是参考，不是对本方法有效性的保证。新增代码没有逐段复制以下作者实现。

1. ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration.
   Paper: https://arxiv.org/html/2505.24357v3
   Authors' code: https://github.com/XIANGLONGYAN/ReCalKV
   Relevant: Section 3.2 (head reordering), 3.3 (offline Value calibration); `palu/model/modules/svd_linear.py`, `palu/decomposition.py` as indexed by repository README.
   Difference: authors group heads for low-rank decomposition. Here a target layer's GQA groups are matched to another layer's already cached groups by actual read error; online KV is not reordered.

2. xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction.
   Paper: https://arxiv.org/html/2503.18893v2
   Code: https://github.com/abdelfattah-lab/xKV
   Relevant: cross-layer common subspaces, iso-compression comparison, selective reconstruction. Strong comparison, not proof that a single original head suffices.

3. KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing.
   Paper: https://arxiv.org/abs/2410.18517
   Relevant: layer-wise physical KV sharing and the limitations of raw similarity as a proxy.

4. FlashInfer: attention states / recursive attention.
   Official docs: https://docs.flashinfer.ai/tutorials/recursive_attention.html
   Relevant: conditional outputs and logsumexp combine with a common normalization. Our code retains the repository's existing vLLM FA2 backend rather than adding this dependency.

5. Palu: Compressing KV-Cache with Low-Rank Projection.
   Paper: https://arxiv.org/abs/2407.21118
   Code: https://github.com/shadowpa0327/Palu
   Relevant: compression alone does not imply latency; efficient kernels/appropriate workload matter.

6. User repository, audited branch/commit:
   https://github.com/utazh/pans/tree/b48c45d779ecde8ae960a6e4837d8b6673c57600
   Audited `sharedkv_v21/runtime.py`, `refine.py`, `ops.py`, `guard.py`, old Pcache bridge, and README_LAYERKV_V21.md; numerical conclusions use the attached EXPERIMENT_REPORT_V21_ZH.md.

This is a focused reference review, not an exhaustive novelty search. No priority/first-of-its-kind claim is made.
