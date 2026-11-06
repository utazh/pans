# Primary sources and code reviewed

Research references are motivations/comparators, not evidence that v2.5 achieves quality or latency gains.

## Source repository

- Audited code: https://github.com/utazh/pans/tree/a6be130c921024a4cd358ccf5841e4a46f547ae2
- Repaired replay: src/contiguous_fuxian/sharedkv_v24/faithful.py
- Candidate-score calibration: src/contiguous_fuxian/sharedkv_v24/train.py and replay.py
- Existing physical source management: sharedkv/runtime.py, sharedkv_v21/runtime.py
- Loader methods: flexgen_pcache.py:_submit_positions, _resolve_entry, metrics

No upstream files are bundled or rewritten. Installed additions depend on the exact interface hashes in guard.py. Repository content was read through the connected GitHub tool. CPU tests use a minimal interface fixture, not a complete remote checkout or server run.

## DeepSeek-V4.1

- Report: https://arxiv.org/pdf/2609.19969
- Official model configuration: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json
- Read sec 2.3.1 and Figure 4 (global KV vs per-layer Q/SWA), sec 3.1.2 (shared-state training), sec 3.2 (persistent cache and bounded replay).
- v2.5 borrows the separation between large shared memory and small layer-specific memory, NOT its model architecture or SWA bounded replay algorithm. Here native prefix patches are loaded from existing SSD KV, not recomputed.

## Related compression/sharing research

- Cross-Layer Attention (CLA): https://arxiv.org/abs/2405.12981
  Adjacent-layer KV sharing trained as an architecture. Shows why cross-layer sharing alone is not an unoccupied novelty claim.
- KVSharer: https://arxiv.org/abs/2410.18517
  Post-hoc layer sharing. Direct-sharing controls should remain in experiments.
- MiniCache: https://arxiv.org/abs/2405.14366
  Cross-layer merging plus retention of distinct token states. Shared memory with native exceptions is not, by itself, a first-ever concept.
- xKV: https://arxiv.org/abs/2503.18893
  Code: https://github.com/abdelfattah-lab/xKV
  Shared subspaces and selective reconstruction are relevant strong baselines; comparable memory/quality accounting is required.
- ReCalKV: https://arxiv.org/abs/2505.24357
  Code: https://github.com/XIANGLONGYAN/ReCalKV
  Head reordering and offline value calibration. v2.2's negative task-level result is retained rather than silently replaced.
- DeepSeek-V2/MLA: https://arxiv.org/abs/2405.04434
  Moving linear transforms to query/output paths is prior art; do not claim the algebra alone as novelty.

## Portability

- Qwen2.5-7B config: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json
- Qwen2.5-14B config: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/raw/main/config.json
- 14B model card (14.7B parameters): https://huggingface.co/Qwen/Qwen2.5-14B-Instruct

14B uses Qwen2, 48 layers, 40 query heads, 8 KV heads, d=128. This is architectural compatibility for the proposed mathematics, not real-model or GPU certification. Larger models require their own cache/index/teacher/readers and adequate memory. Current code has no TP/device-map implementation.

## Benchmark semantics

- https://docs.kernel.org/admin-guide/sysctl/vm.html
- https://man7.org/linux/man-pages/man2/posix_fadvise.2.html
- https://man7.org/linux/man-pages/man2/mincore.2.html
- https://docs.pytorch.org/docs/2.10/checkpoint.html

The published v23 cold protocol is reused unchanged. Model/reader warmup and target payload coldness are different. No global drop_caches, no claimed SSD firmware-cache eviction, no profiler data substituted for formal latency.
