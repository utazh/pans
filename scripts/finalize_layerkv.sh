#!/bin/bash
set -euo pipefail
cd /home/panzihang/src/prism_layerkv
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
PY=/home/panzihang/venvs/vllm-stable/bin/python
$PY scripts/run_layerkv_suite.py configs/layerkv_fixed_verification.json > logs/fixed_verification.log 2>&1
cp logs/final_tests.log logs/final_tests_before_fixture_copy.log
$PY -m pytest tests -q > logs/final_tests.log 2>&1
$PY scripts/analyze_layerkv.py > logs/final_analysis.log 2>&1
$PY scripts/check_layerkv_stress.py > logs/stress_check.log 2>&1
$PY scripts/write_layerkv_report.py > logs/write_report.log 2>&1
echo completed > logs/finalization.done
