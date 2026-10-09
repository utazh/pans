#!/bin/bash
# 7B SST-2 ridge-only 补跑：等空闲显存 >=30 GB 的卡，失败（如他人/自身任务抢占显存导致 OOM）则重试，最多 3 次。
B=/home/shensicheng/qyd/pans; source $B/env_q7.sh; ulimit -n 1048576; cd $B/src
for try in 1 2 3; do
  until g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '($1==4||$1==5||$1==6)&&$2>30000{print $1; exit}') && [ -n "$g" ]; do sleep 120; done
  echo "try $try gpu $g $(date)"
  $PANS_PY scripts/v29m_pipeline.py all --cells sst2_b25 --gpu $g --cores 112-115 --steps 0 --tag ridge && { echo RERUN_OK; exit 0; }
  sleep 300
done
echo RERUN_FAILED
