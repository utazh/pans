#!/bin/bash
# 等全部基线质量评估结束（避免磁盘干扰）后，在 GPU 4 依次跑 7B/14B/32B 的 4 系统同块计时（P1-c 设计）。
B=/home/shensicheng/qyd/pans
sleep 60
while ps -u shensicheng -o args | grep -q "[s]ysquality.sh"; do sleep 300; done
sleep 60
for T in q7 q14 q32; do
  if [ $T = q7 ]; then R=$B/src; export PANS_MODEL_PATH=$B/models/Qwen2.5-7B-Instruct PANS_SELECTOR_KV_HEAD_IDS=0,1,2,3 PANS_PY=$B/venv/bin/python;
  else R=$B/src_$T; . $B/env_$T.sh; fi
  cd $R && . $B/env_timing.sh && export PANS_ROOT=$R
  $B/venv/bin/python scripts/v29m_timing_sys.py a100_sys_$T --model-tag $T --tag s800 > $B/logs/a100_sys_$T.log 2>&1
done
echo SYS_TIMING_ALL_DONE
