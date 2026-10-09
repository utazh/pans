#!/bin/bash
# warm 计时结束后补跑 Llama ridge-only SUBJ 的测试阶段（S3 先前 OOM 失败）。
B=/home/shensicheng/qyd/pans
until grep -q L8_WARM_DONE $B/logs/queue_l8_warm.log; do sleep 120; done
source $B/env_l8.sh; ulimit -n 1048576; cd $B/src_l8
until g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '($1==4||$1==5||$1==6)&&$2>30000{print $1; exit}') && [ -n "$g" ]; do sleep 120; done
$PANS_PY scripts/v29m_pipeline.py test --cells subj_b25 --gpu $g --tag ridge --reuse-from v29m_s800 && $PANS_PY $B/analysis/gen_summary.py > $B/analysis/reader_sources.txt
echo AFTER_WARM_DONE
