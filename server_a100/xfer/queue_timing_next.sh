#!/bin/bash
# 7B 计时完成后，依次在 GPU 6 跑 14B、32B 冷启动计时（同一计时环境；每个模型用自己的 worktree 与 pcache 副本）。
B=/home/shensicheng/qyd/pans
until [ -f $B/src/studies/v29/a100_timing_q7/done.json ]; do sleep 300; done
while pgrep -u shensicheng -f "scripts/v29m_pipeline.py" > /dev/null; do sleep 120; done
for T in q14 q32; do
  cd $B/src_$T && . $B/env_timing.sh && . $B/env_$T.sh
  $B/venv/bin/python scripts/v29m_timing.py a100_timing_$T --tag s800 > $B/logs/a100_timing_$T.log 2>&1
done
echo ALL_TIMING_DONE
