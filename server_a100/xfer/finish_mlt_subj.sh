#!/bin/bash
# SUBJ 留一的检查点已存在（16:11，800 步，best_step 700）；补完成标记后跑 derive/dev/test（等空闲 GPU，失败重试一次）。
B=/home/shensicheng/qyd/pans; source $B/env_l8.sh; ulimit -n 1048576; cd $B/src_l8
W2=studies/v29m_mlt; T=subj_b25
cp studies/v29m_s800/plans/${T}_plan.json $W2/plans/; for m in capture_fit capture_val; do cp studies/v29m_s800/done/${T}_$m $W2/done/; done
echo '{"model_level": true, "train_cells": "sst2_b25,trec_b25,rte_b25", "held_out": "subj_b25", "steps": 800}' > $W2/done/${T}_train
pick() { until g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '($1==4||$1==5||$1==6)&&$2>30000{print $1; exit}') && [ -n "$g" ]; do sleep 120; done; }
for st in derive dev test; do
  pick; $PANS_PY scripts/v29m_pipeline.py $st --cells $T --gpu $g --tag mlt --reuse-from v29m_s800 || { sleep 300; pick; $PANS_PY scripts/v29m_pipeline.py $st --cells $T --gpu $g --tag mlt --reuse-from v29m_s800; } || { echo STAGE_FAILED $st; exit 1; }
done
echo MLT_DONE $T
