#!/bin/bash
set -euo pipefail
cd /home/panzihang/src/prism_layerkv
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=/home/panzihang/venvs/vllm-stable/bin/python
while ! grep -q GPU2_QUALITY_COMPLETE logs/v21_gpu2_next.log || ! grep -q GPU3_QUALITY_COMPLETE logs/v21_gpu3_next.log || [ ! -f profiles/v21/viz_hybrid_shared_retry/validated.json ]; do
 if grep -qE 'Traceback|QUALITY_FAILED|SEQUENTIAL_FAILED' logs/v21_gpu2_next.log logs/v21_gpu3_next.log; then echo DEPENDENCY_FAILED; exit 1; fi
 sleep 10
done
echo FORMAL_TIMING_BEGIN
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv -l 1 > logs/v21_timing_gpu.csv &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
$PY scripts/run_layerkv_v21_suite.py scripts/v21_timing_jobs.json
kill "$monitor_pid" 2>/dev/null || true
trap - EXIT
for tokens in 1 41 49; do
 $PY scripts/bench_sharedkv_v21.py --backend vllm2 --adapter assets/v21_refit27_s200/trec.pt --consumer 27 --selected-tokens 1228 --query-tokens "$tokens" --iterations 80 --warmup 20 --output results/v21_micro_q"$tokens".json
done
$PY scripts/analyze_layerkv_v21.py --prefix v21_reg_ --baseline v21_reg_identity --output results/v21_reg_compare.json
$PY scripts/analyze_layerkv_v21.py --prefix v21_hold_ --baseline v21_hold_original --output results/v21_hold_compare.json
$PY scripts/analyze_layerkv_v21.py --prefix v21_fixed_ --baseline v21_fixed_shared --output results/v21_fixed_compare.json
echo ALL_RUNS_COMPLETE
