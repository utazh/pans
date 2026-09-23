# Primary sources and inspected code

## Actual audited project

- utazh/pans v2.5, commit `5e523ec7f48b99b54cf449ace7b8f95e94c7e8db`.
- Report: `EXPERIMENT_REPORT_V25_ZH.md` (user attachment; 249 parsed lines).
- Code paths inspected through the GitHub connector:
  - `src/contiguous_fuxian/sharedkv_v25/{policy,attention,runtime,replay,train}.py`
  - `scripts/run_native_prefix_v25.py`
  - `scripts/run_v25_experiment_pipeline.py`
  - `src/contiguous_fuxian/flexgen_pcache.py` (resolve and score/admission boundary)
  - `vendor/flexgen/my_pcache_fast.py` (disk reads and cache movement)
- The package contains new controls, not a copied replacement of the user's project. Existing implementation notices and licenses remain unchanged.

## Closest related work (not proofs of our method)

- Brandon et al., *Reducing Transformer Key-Value Cache Size with Cross-Layer Attention*, 2024. https://arxiv.org/abs/2405.12981
  Adjacent-layer architectural sharing, including from-scratch training. Cross-layer sharing by itself is not novel.
- Yang et al., *KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing*, 2024. https://arxiv.org/abs/2410.18517
  Post-training layer sharing and source selection. Our direct identity-reader ablation is not a reproduction of its complete method.
- Liu et al., *MiniCache: KV Cache Compression in Depth Dimension for Large Language Models*, 2024. https://arxiv.org/html/2405.14366
  Inter-layer merging with token retention and restoration. A generic shared-plus-native idea is not new.
- Chang et al., *xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction*, v2 (2026). https://arxiv.org/html/2503.18893v2
  Grouped-layer factorization and Selective Reconstruction. Must compare current SR variants, not only expensive full reconstruction.
  Author code: https://github.com/abdelfattah-lab/xKV
- Yan et al., *ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration*, 2025. https://arxiv.org/html/2505.24357v3
  Asymmetric K/V treatment and offline value calibration. Author code: https://github.com/XIANGLONGYAN/ReCalKV
- DeepSeek-AI, *DeepSeek-V2*, 2024. https://arxiv.org/abs/2405.04434
  MLA and projection absorption: matrix reassociation should not be claimed as our invention.

## Statistical rule

- SciPy BinomTestResult.proportion_ci documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html
  Exact binomial intervals. Our script uses the existing v25 Bonferroni composition of harmful/beneficial bounds, not a newly chosen rule. The result remains diagnostic and does not auto-promote a checkpoint.

## Important limitation

The local environment could not clone GitHub directly. Actual repository text was read via the connected GitHub tools. CPU mathematical dependencies were assembled from the conversation-provided prototype archives; consumed v25 modules and the reconstructed current sharedkv/core.py were checked against the actual Git blob hashes. No full source-tree, real Qwen model or CUDA regression is claimed.
