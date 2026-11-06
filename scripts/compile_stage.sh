#!/bin/bash
set -euo pipefail
cd /home/panzihang/src/prism_layerkv
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=/home/panzihang/venvs/vllm-stable/bin/python
$PY scripts/compile_sharedkv.py --store-root runtime_data/prefix_store --task trec --fit-traces runtime_data/cal_fit --validation-traces runtime_data/cal_val --pairs 2:3,6:7,10:11,14:15,18:19,22:23,26:27 --key-mode dense --output assets/layerkv_strict_forward/trec.pt --diagnostic-output assets/layerkv_diag_forward/trec.pt > logs/compile_forward.log 2>&1
$PY scripts/compile_sharedkv.py --store-root runtime_data/prefix_store --task trec --fit-traces runtime_data/cal_fit --validation-traces runtime_data/cal_val --pairs 3:2,7:6,11:10,15:14,19:18,23:22,27:26 --key-mode dense --output assets/layerkv_strict_reverse/trec.pt --diagnostic-output assets/layerkv_diag_reverse/trec.pt > logs/compile_reverse.log 2>&1
echo completed > logs/compile.done
