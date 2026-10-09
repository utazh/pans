#!/bin/bash
# G3 一次留一法：等空闲 GPU（>=30 GB）→ 多前缀训练（其余 3 个任务）→ 导出给目标任务 → 流水线 derive/dev/test。
# 用法: mlt_run.sh <worktree> <envfile> <src_tag> <target_cell> <train_cells> <steps> <out_tag>
B=/home/shensicheng/qyd/pans; WT=$1; ENVF=$2; SRC=$3; T=$4; TRAIN=$5; STEPS=$6; OUT=$7
source $B/$ENVF; ulimit -n 1048576; cd $B/$WT
free_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '($1==4||$1==5||$1==6)&&$2>30000{print $1; exit}'; }
pick() { until g=$(free_gpu) && [ -n "$g" ]; do sleep 120; done; }; pick
echo "target $T train $TRAIN gpu $g $(date)"
W2=studies/v29m_$OUT; task=${T%_b*}
mkdir -p $W2/done $W2/plans
if [ ! -f $W2/done/${T}_train ]; then
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
    $PANS_PY $B/xfer/train_multi_v24.py --model-path $PANS_MODEL_PATH --w studies/v29m_$SRC --train-cells $TRAIN --target $T \
    --out-w $W2 --steps $STEPS || { sleep 300; pick; echo "retry gpu $g"; CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src $PANS_PY $B/xfer/train_multi_v24.py --model-path $PANS_MODEL_PATH --w studies/v29m_$SRC --train-cells $TRAIN --target $T --out-w $W2 --steps $STEPS; } || { echo TRAIN_FAILED; exit 1; }
  cp studies/v29m_$SRC/plans/${T}_plan.json $W2/plans/
  for m in capture_fit capture_val; do cp studies/v29m_$SRC/done/${T}_$m $W2/done/; done
  echo "{\"model_level\": true, \"train_cells\": \"$TRAIN\", \"held_out\": \"$T\", \"steps\": $STEPS}" > $W2/done/${T}_train
fi
for st in derive dev test; do
  $PANS_PY scripts/v29m_pipeline.py $st --cells $T --gpu $g --tag $OUT --reuse-from v29m_$SRC || { sleep 300; pick; $PANS_PY scripts/v29m_pipeline.py $st --cells $T --gpu $g --tag $OUT --reuse-from v29m_$SRC; } || { echo "STAGE_FAILED $st"; exit 1; }
done
echo MLT_DONE $T
