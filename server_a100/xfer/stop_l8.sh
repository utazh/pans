#!/bin/bash
# 只停我们自己的 Llama 相关进程：重启/排队脚本 + src_l8 流水线（按进程组整组停）。
pkill -u shensicheng -f "[r]elaunch_l8.sh"; pkill -u shensicheng -f "[q]ueue_after_l8.sh"
for pid in $(pgrep -u shensicheng -f "[v]29m_pipeline.py all --cells"); do
  cwd=$(readlink /proc/$pid/cwd)
  if [ "$cwd" = /home/shensicheng/qyd/pans/src_l8 ]; then
    pg=$(ps -o pgid= -p $pid | tr -d " "); echo "stop pipeline $pid pgid $pg"; kill -TERM -- -$pg
  fi
done
sleep 5
ps -u shensicheng -o pid,pgid,etime,args | grep -E "[v]29m_pipeline|[t]rain_instrumented|[r]un_evidence|[r]elaunch|[q]ueue_" | cut -c1-140
echo STOPPED
