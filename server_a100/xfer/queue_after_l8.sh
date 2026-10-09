#!/bin/bash
# Llama 4 个流水线全部结束后（磁盘才安静）：Llama 计时准备 → Llama 4 系统冷计时 → P2 并发正式实验。
B=/home/shensicheng/qyd/pans
until grep -q L8_PIPELINES_LAUNCHED $B/logs/queue_l8.log; do sleep 60; done
sleep 120
while pgrep -u shensicheng -f "[v]29m_pipeline.py all" > /dev/null; do sleep 120; done
echo "pipelines finished $(date)"; grep -l -E "Traceback|SystemExit" $B/logs/l8_pipe_*.log && echo "WARN: some pipeline log has errors"

if bash $B/xfer/prep_l8_timing.sh 0 > $B/logs/prep_l8_timing.log 2>&1; then
  ( cd $B/src_l8
    . $B/env_timing.sh; . $B/env_l8.sh; export PANS_ROOT=$B/src_l8      # 顺序：先计时环境，再模型环境
    echo "l8 model=$PANS_MODEL_PATH heads=$PANS_SELECTOR_KV_HEAD_IDS root=$PANS_ROOT"
    $B/venv/bin/python scripts/v29m_timing_sys.py a100_sys_l8 --model-tag l8 --tag s800 --conds preserve > $B/logs/a100_sys_l8.log 2>&1 )
  echo "l8 timing done $(date)"
else
  echo "PREP_L8_TIMING_FAIL $(date) — 跳过 Llama 计时，继续并发实验"
fi

cd $B/src
. $B/env_timing.sh                                                   # 7B
$B/venv/bin/python $B/xfer/conc.py main --cells trec_b25,sst2_b25 --arms I,SEL,CKV --ns 1,2,4,8 --reps 2 --gpus 0,1,2,4 \
   > $B/logs/conc_main.log 2>&1
echo "ALL_AFTER_L8_DONE $(date)"
