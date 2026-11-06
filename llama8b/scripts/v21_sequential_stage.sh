#!/bin/bash
set -euo pipefail
cd /home/panzihang/src/prism_layerkv
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=/home/panzihang/venvs/vllm-stable/bin/python
$PY scripts/run_layerkv_v21_suite.py scripts/v21_student2_jobs.json
$PY scripts/refine_sharedkv_v21.py --store-root runtime_data/prefix_store --task trec --teacher-fit runtime_data/v21_teacher_fit --teacher-val runtime_data/v21_teacher_val --student-fit runtime_data/v21_student2_fit --student-val runtime_data/v21_student2_val --source 18 --target 19 --initial-reader assets/layerkv_dense7/trec.pt --carry assets/v21_refit2_s200/trec.pt --steps 200 --eval-every 20 --lr 0.0001 --device cuda --bf16-ste --output assets/v21_seq2/trec.pt
$PY scripts/run_layerkv_v21_suite.py scripts/v21_student19_jobs.json
$PY scripts/refine_sharedkv_v21.py --store-root runtime_data/prefix_store --task trec --teacher-fit runtime_data/v21_teacher_fit --teacher-val runtime_data/v21_teacher_val --student-fit runtime_data/v21_student19_fit --student-val runtime_data/v21_student19_val --source 26 --target 27 --initial-reader assets/layerkv_dense7/trec.pt --carry assets/v21_seq2/trec.pt --steps 200 --eval-every 20 --lr 0.0001 --device cuda --bf16-ste --output assets/v21_seq3/trec.pt
$PY scripts/run_layerkv_v21_suite.py scripts/v21_refit_quality_jobs.json
echo SEQUENTIAL_COMPLETE
