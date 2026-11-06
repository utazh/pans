#!/bin/bash
set -euo pipefail
cd /home/panzihang/src/prism_layerkv
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=/home/panzihang/venvs/vllm-stable/bin/python
$PY scripts/run_sharedkv.py --mode original --logits results/original_smoke_logits -- --task trec --budget 25 --period 8 --samples 16 --warmup 0 --bundle-dir data/sharedkv_splits/test --output results/original_smoke > logs/original_smoke.log 2>&1
$PY scripts/run_sharedkv.py --mode identity --backend vllm2 --logits results/identity_smoke_logits --stats results/identity_smoke.extra.jsonl -- --task trec --budget 25 --period 8 --samples 16 --warmup 0 --bundle-dir data/sharedkv_splits/test --output results/identity_smoke > logs/identity_smoke.log 2>&1
$PY scripts/run_sharedkv.py --mode collect --trace-dir runtime_data/cal_fit --max-query-samples 8 -- --task trec --budget 25 --period 8 --samples 32 --warmup 0 --bundle-dir data/sharedkv_splits/fit --output results/collect_fit > logs/collect_fit.log 2>&1
$PY scripts/run_sharedkv.py --mode collect --trace-dir runtime_data/cal_val --max-query-samples 8 -- --task trec --budget 25 --period 8 --samples 32 --warmup 0 --bundle-dir data/sharedkv_splits/val --output results/collect_val > logs/collect_val.log 2>&1
echo completed > logs/stage1.done
