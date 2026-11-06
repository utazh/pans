#!/bin/bash
set -euo pipefail
cd /home/panzihang/src/prism_layerkv
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
while ! grep -q SEQUENTIAL_COMPLETE logs/v21_sequential.log; do
 if grep -q Traceback logs/v21_sequential.log; then echo SEQUENTIAL_FAILED; exit 1; fi
 sleep 10
done
/home/panzihang/venvs/vllm-stable/bin/python scripts/run_layerkv_v21_suite.py scripts/v21_holdout_gpu3_jobs.json
echo GPU3_QUALITY_COMPLETE
