#!/bin/bash
set -euo pipefail
cd /home/panzihang/src/prism_layerkv
while kill -0 50714 2>/dev/null; do sleep 5; done
grep -q 'SUITE COMPLETE' logs/baselines.log
exec env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /home/panzihang/venvs/vllm-stable/bin/python scripts/run_layerkv_suite.py configs/layerkv_experiments.json
