# Primary references and inspected code

## User repository (audited revision)

- Branch/commit: https://github.com/utazh/pans/tree/fddabb7015c66c870a11810069bfbe20d5a34bbf
- Latest results: https://github.com/utazh/pans/blob/fddabb7015c66c870a11810069bfbe20d5a34bbf/EXPERIMENT_REPORT_V22_ZH.md
- Group-compatible runtime and selector/ownership: `src/contiguous_fuxian/sharedkv_v21/runtime.py`, `src/contiguous_fuxian/sharedkv/io.py`.
- Existing calibration: `src/contiguous_fuxian/sharedkv_v21/refine.py` and v22 modules.
- Exact budget: `allocate_exact_layer_blocks` in `flexgen_qwen_reprefill.py`, `keep_blocks_for_layer` in `flexgen_pcache.py`.
- Real storage objects: `PrefixKVLayer`, `ChunkCache`, `ScoreUnit`, `Pcache` in `vendor/flexgen/my_pcache_fast.py`; actual token.path is the cold target.
- Read advice: vendor `uncache` is POSIX_FADV_DONTNEED, invoked before disk memmaps in `_chunk_source_readonly` (around lines 1490–1540). The page-hot control must account for this.
- Serial queue: `src/contiguous_fuxian/priority_executor.py`.

## Research inspiration (not evidence this proposal works)

1. BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction.
   Paper: https://arxiv.org/abs/2102.05426
   Author code: https://github.com/yhhhli/BRECQ/blob/main/quant/block_recon.py
   Use: block-level reconstruction rather than isolated operator error. Original experiments are quantization/CV, not frozen-Qwen cross-layer KV sharing.
2. GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration.
   Paper: https://arxiv.org/abs/2504.02692
   Author code: https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ
   Use: actual perturbed student inputs matched to clean teacher outputs. We do not reuse the quantization update formula.
3. ReCalKV: https://arxiv.org/abs/2505.24357 ; https://github.com/XIANGLONGYAN/ReCalKV
   Existing reference for asymmetric K/V calibration/head organization. v2.2 user's results, not this paper, determine that head routing is not the present main line.
4. xKV: https://arxiv.org/abs/2503.18893 ; https://github.com/abdelfattah-lab/xKV
   Strong comparison if a single original source's feature space proves inadequate; common-subspace construction is not implemented in this small patch.

## Linux contracts used by cold experiments

- POSIX_FADV_DONTNEED advisory semantics, dirty/partial pages: https://man7.org/linux/man-pages/man2/posix_fadvise.2.html
- mincore residency snapshot: https://man7.org/linux/man-pages/man2/mincore.2.html
- process I/O accounting vs syscall bytes: https://man7.org/linux/man-pages/man5/proc_pid_io.5.html
- drop_caches global effects and warnings: https://docs.kernel.org/admin-guide/sysctl/vm.html#drop-caches

The new code is independently authored for this integration. It imports the user's existing modules and PyTorch; it does not vendor BRECQ/GPTAQ source or copy their implementations. Existing upstream licenses/provenance remain unchanged. All linked material is a reference, not a claimed author endorsement or performance guarantee.
