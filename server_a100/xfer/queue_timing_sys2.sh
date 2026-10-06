#!/bin/bash
# 14B/32B 4 系统计时（修正：先加载计时环境，再加载模型环境，避免 env_timing.sh 中的 7B 模型路径覆盖）。
B=/home/shensicheng/qyd/pans
for T in q14 q32; do
  R=$B/src_$T; cd $R
  . $B/env_timing.sh; . $B/env_$T.sh; export PANS_ROOT=$R
  echo "$T model=$PANS_MODEL_PATH heads=$PANS_SELECTOR_KV_HEAD_IDS root=$PANS_ROOT"
  $B/venv/bin/python scripts/v29m_timing_sys.py a100_sys_$T --model-tag $T --tag s800 > $B/logs/a100_sys_$T.log 2>&1
done
echo SYS_TIMING_Q14Q32_DONE
