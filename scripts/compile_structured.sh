#!/bin/bash
set -euo pipefail
cd /home/panzihang/src/prism_layerkv
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=/home/panzihang/venvs/vllm-stable/bin/python
for mode in rope rope_rr; do
$PY scripts/compile_sharedkv.py --store-root runtime_data/prefix_store --task trec --fit-traces runtime_data/cal_fit --validation-traces runtime_data/cal_val --pairs 26:27 --key-mode "$mode" --rank 16 --output "assets/layerkv_${mode}_strict/trec.pt" --diagnostic-output "assets/layerkv_${mode}_diagnostic/trec.pt" > "logs/compile_${mode}.log" 2>&1
done
