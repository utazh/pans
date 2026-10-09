#!/bin/bash
# 等我们自己的质量评测/训练全部结束（磁盘、GPU 4 安静）后，跑 Llama 4 系统 warm 计时（25%，6 块 × 16 请求）。
B=/home/shensicheng/qyd/pans
sleep 600
while pgrep -u shensicheng -f "[v]29m_pipeline.py|[t]rain_multi_v24|[m]lt_run2|[r]erun_7b_sst2|[q]ueue_mlr" > /dev/null; do sleep 300; done
echo "start warm $(date)"
cd $B/src_l8
. $B/env_timing.sh; . $B/env_l8.sh; export PANS_ROOT=$B/src_l8
$B/venv/bin/python scripts/v29m_timing_sys.py a100_sys_l8_warm --model-tag l8 --tag s800 --conds warm
$B/venv/bin/python $B/xfer/timing_report.py studies/v29/a100_sys_l8_warm warm > $B/analysis/a100_sys_l8_warm.json
echo L8_WARM_DONE
