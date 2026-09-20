#!/bin/bash
set -euo pipefail
cd /home/panzihang/src/prism_layerkv
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
while ! grep -q GPU3_QUALITY_COMPLETE logs/v21_gpu3_next.log; do
 if grep -q Traceback logs/v21_gpu3_next.log; then exit 1; fi
 sleep 10
done
/home/panzihang/venvs/vllm-stable/bin/python scripts/run_layerkv_v21_suite.py scripts/v21_hybrid_seq3_holdout_jobs.json
/home/panzihang/venvs/vllm-stable/bin/python scripts/profile_layerkv_v21.py --profiler viztracer --profile-output profiles/v21/viz_hybrid_shared_retry -- --mode shared --diagnostic --nvtx --layout hybrid --metadata-cache --packed-ops --adapters assets/v21_refit27_s200 -- --task trec --budget 25 --period 8 --samples 8 --warmup 4 --bundle-dir data/sharedkv_splits/test --output profiles/v21/viz_hybrid_shared_retry_run
echo VIZ_RETRY_COMPLETE
