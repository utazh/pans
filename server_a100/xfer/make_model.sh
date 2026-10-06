#!/bin/bash
# 为一个新模型建 worktree + 基础产物。用法: make_model.sh <tag> <ModelDir> <gpu> <probe_q_heads> <selector_kv_heads>
#   make_model.sh q14 Qwen2.5-14B-Instruct 4 0,5,10,15,20,25,30,35 0,1,2,3,4,5,6,7
set -euo pipefail
B=/home/shensicheng/qyd/pans; T=$1; M=$2; G=$3; PQH=$4; SKV=$5; WT=$B/src_$T
cd $B/src
[ -d $WT ] || git worktree add -q -b a100-$T $WT a100
cd $WT
ln -sfn $B/src/data data; mkdir -p studies; ln -sfn $B/src/studies/v28_frozen studies/v28_frozen; ln -sfn $B/src/audit audit
bash $B/xfer/build_base.sh $M $T $G $PQH $SKV
mkdir -p runtime_data assets
ln -sfn $B/work/$T/prefix_store runtime_data/prefix_store
ln -sfn $B/work/$T/pcache_c16 runtime_data/pcache_c16
ln -sfn $B/work/$T/selector_index_k4_g32 assets/selector_index_k4_g32
echo "export PANS_MODEL_PATH=$B/models/$M PANS_PY=$B/venv/bin/python PANS_PROBE_QUERY_HEADS=$PQH PANS_SELECTOR_KV_HEAD_IDS=$SKV TMPDIR=$B/tmp" > $B/env_$T.sh
echo MODEL_READY_$T
