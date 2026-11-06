#!/bin/bash
set -euo pipefail
cd /home/panzihang/src/prism_layerkv
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=/home/panzihang/venvs/vllm-stable/bin/python
while ! grep -q ALL_RUNS_COMPLETE logs/v21_formal.log; do
 if grep -q Traceback logs/v21_formal.log; then echo FORMAL_FAILED; exit 1; fi
 sleep 10
done
$PY scripts/bench_sharedkv_v21.py --backend vllm2 --adapter assets/v21_refit27_s200/trec.pt --consumer 27 --selected-tokens 1228 --query-tokens 1 --private-tokens 42 --iterations 80 --warmup 20 --output results/v21_micro_decode_q1_tail42.json
$PY scripts/check_v21_invariants.py
$PY scripts/analyze_v21_timing.py
$PY scripts/write_layerkv_v21_report.py
$PY scripts/finalize_v21_delivery.py
runtime_tools/nsight-2025.5.1/opt/nvidia/nsight-systems-cli/2025.5.1/target-linux-x64/nsys --version
$PY -c "import sys;sys.path.insert(0,'runtime_deps');import viztracer;print('isolated VizTracer',viztracer.__file__)"
echo V21_FINISHED
