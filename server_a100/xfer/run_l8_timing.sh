#!/bin/bash
# Llama 4 系统 SSD 冷计时（25%，6 块 × 16 请求，Williams 顺序）。先计时环境，再模型环境。
B=/home/shensicheng/qyd/pans
cd $B/src_l8
. $B/env_timing.sh; . $B/env_l8.sh; export PANS_ROOT=$B/src_l8
echo "l8 model=$PANS_MODEL_PATH heads=$PANS_SELECTOR_KV_HEAD_IDS root=$PANS_ROOT gpu=$PANS_TIMING_GPU"
$B/venv/bin/python scripts/v29m_timing_sys.py a100_sys_l8 --model-tag l8 --tag s800 --conds preserve
echo L8_TIMING_DONE
