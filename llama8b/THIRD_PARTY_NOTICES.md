# Third-party provenance

The existing `vendor/flexgen` tree contains FlexGen-derived utilities and project-specific Pcache / HyperInfer-style experimental code. It is retained from the user's experimental tree; it is not presented as an unmodified upstream checkout.

The FlexGen project is now hosted as [FMInference/FlexLLMGen](https://github.com/FMInference/FlexLLMGen). Its [upstream Apache-2.0 license](https://github.com/FMInference/FlexLLMGen/blob/main/LICENSE), including the upstream copyright notice, is reproduced in `licenses/FlexLLMGen-LICENSE`. Existing source notices are retained.

Local changes documented by the experiment reports include asynchronous prefetch handling, bounded pinned staging, GPU gather/scatter, host batch gather and metadata reuse. In particular, `vendor/flexgen/my_pcache_fast.py` is a locally modified experimental implementation. Baseline copies under `audit/` retain earlier versions for regression testing.

The experimental input did not contain a top-level license declaration. This publication does not assign a blanket license to all project-specific additions or claim that every vendored file has identical provenance. Model weights, datasets and runtime tool distributions are not included.
