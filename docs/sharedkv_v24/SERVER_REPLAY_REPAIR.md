# v2.4 server replay repair

The supplied v24 all-layer reference replay failed its unchanged native parity gate on 64 real Qwen2.5-7B validation requests: maximum mean-candidate score error 0.6223239898681641, maximum token log-probability error 0.6848239898681641. Thresholds remained 0.02 / 0.10. Training stopped before any update; rejected report and logs remain in assets/v24_smoke and logs/v24_smoke.log.

Cause localized by comparing the replay with the actual hybrid packed scorer: the reference used full-sequence mathematical split attention while inference uses vLLM FA2 native concatenated attention and incremental candidate scoring. The server calibration replay now uses the existing FA2 forward, existing packed prepare_query/finish, actual prefill/incremental LM-head shapes, and functionally threaded layer-private current-token KV. Only new v24 files were changed. The existing inference kernel, cache management, cold certification and all guarded old interfaces remain unchanged.

FA2 forward lacks the required autograd path through this wrapper. AttentionVJP implements the mathematical softmax vector-Jacobian product in FP32, including gradients through LSE used by the prefix/private merge. BF16 effective reader rounding uses the existing straight-through estimator. This backward is a mathematical surrogate for finite-precision FA2 arithmetic, not a claim of differentiating quantization itself.

Validation: 48 package/server tests passed, including CPU double-precision gradchecks, CUDA exact FA2 forward equality and reference VJP comparisons. A real-model one-step smoke completed with native score and token errors both exactly zero on all 64 validation requests, and nonzero reader optimization. Peak allocated memory was 15,732,978,688 bytes. Initial A/B/bk/bv tensor SHA256 values match all three v23 group-control initializations exactly (audit/v24_matched_initialization.json). Smoke outputs are not formal checkpoints.

Synthetic CUDA records without a teacher_backend field retain the supplied mathematical reference path; captured production rows dispatch to the faithful path and unsupported backend/layout combinations fail explicitly. This preserves the package toy-test contract rather than falsely tagging toys as audited production data.

Scipy 1.15.3 was installed without dependencies under runtime_deps_v24 solely for the supplied conservative paired quality bound; the shared Python environment was not upgraded.

Formal quality/TTFT and expansion admission are pending. None of the above establishes new accuracy or acceleration.
