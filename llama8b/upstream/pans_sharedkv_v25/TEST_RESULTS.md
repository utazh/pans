# Actual validation of this package

## Executed here

```
36 passed, 1 skipped in 5.43s
```

The command ran `tests/test_sharedkv_v25.py` against a minimal CPU interface fixture. Current v2.4 helper source contents were checked against Git blob hashes; the fixture contains those helpers plus mock original-loader/runner interfaces. It is NOT a full clone of the repository, a full existing-suite result, or a real Qwen run.

Also executed: Python compileall for package source/scripts/tests, successfully.

## Coverage

- Complete selected-chunk partition, shortened final chunks, noncontiguous selection, invalid IDs and oversize repair budgets.
- Disjoint C/R union exactly equals original IDs; no added/dropped prefix support.
- Float64 reference agreement with explicit approximate remote KV plus native patch and private tail, including bias/mass handling and GQA.
- Packed/unpacked reader paths; causal prefill/decode and private-cache forks.
- Source prefix storage is used by view; no full consumer reconstruction.
- Mock loader requests only native patch IDs; admissions/scoring use patch IDs only and preserve logical exact-budget selections.
- Pending-request drainage on error.
- R=0 agrees with the published faithful CPU replay.
- Toy four-layer model: multi-consumer gradients, frozen backbone, activation checkpointing, complete candidate scoring.
- Streamed branch gradient equivalence.
- Training patch restoration and diagnostic new-format export.
- Toy-model online decoder vs differentiable replay, prefill and decode.
- One-step toy calibration, native parity, evaluate/export path.
- Qwen2.5-7B/14B geometry and automatic representative-head derivation (configuration only).

## Skipped

One actual CUDA/vLLM FA2 test with 28 Q heads, 4 KV heads, d=128 was skipped because no CUDA device is available. It must be run on the server.

## Not established

- No true Qwen2.5-7B or 14B quality result for v2.5.
- No RTX 3090 memory or speed measurement.
- No server Pcache/SSD coldness or hardware byte certification here.
- No claim that the existing 510-test suite has been rerun here.
- No end-to-end noninferiority, deployment acceptance, or generalization claim.

Before performance experiments, run the existing entire repository suite, real-model R=0 bridging, R>0 training/inference parity, alias/private-cache checks, and inherited cold precondition tests. Keep failures; do not weaken guards/tolerances to obtain a run.
