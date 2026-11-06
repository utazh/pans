#!/bin/bash
set -euo pipefail
cd /home/panzihang/src/prism_layerkv
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=/home/panzihang/venvs/vllm-stable/bin/python
while [ ! -f results/v21_teacher_val/summary.json ]; do sleep 10; done
for pair in 3:2 18:19 26:27; do
 source=${pair%:*}
 target=${pair#*:}
 for steps in 0 200; do
  $PY scripts/refine_sharedkv_v21.py --store-root runtime_data/prefix_store --task trec --teacher-fit runtime_data/v21_teacher_fit --teacher-val runtime_data/v21_teacher_val --student-fit runtime_data/v21_teacher_fit --student-val runtime_data/v21_teacher_val --source $source --target $target --initial-reader assets/layerkv_dense7/trec.pt --steps $steps --eval-every 20 --lr 0.0001 --device cuda --bf16-ste --output assets/v21_refit${target}_s${steps}/trec.pt
 done
done
echo REFINE_COMPLETE
