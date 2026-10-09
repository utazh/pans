#!/bin/bash
# 打几何回退补丁后重启 Llama 流水线（capture 已完成的单元从 train 继续）；全部重启后再挂上后续计时/并发队列。
B=/home/shensicheng/qyd/pans
$B/venv/bin/python $B/xfer/patch_l8_geometry.py $B/src_l8 || exit 1
source $B/env_l8.sh; ulimit -n 1048576; cd $B/src_l8
CELLS=(sst2_b25 subj_b25 trec_b25 rte_b25); GPUS=(0 1 2 4)
pending="0 1 2 3"
while [ -n "$pending" ]; do
  left=""
  for i in $pending; do
    c=${CELLS[$i]}
    if pgrep -u shensicheng -f "[v]29m_pipeline.py all --cells $c " > /dev/null; then left="$left $i"; continue; fi
    lo=$((64 + 8 * i)); hi=$((lo + 7))
    echo "relaunch $c $(date)"; mv $B/logs/l8_pipe_$c.log $B/logs/l8_pipe_$c.try1.log
    setsid nohup $PANS_PY scripts/v29m_pipeline.py all --cells $c --gpu ${GPUS[$i]} --cores $lo-$hi --steps 800 --tag s800 \
       > $B/logs/l8_pipe_$c.log 2>&1 < /dev/null &
    sleep 20
  done
  pending=$left; [ -n "$pending" ] && sleep 30
done
echo L8_RELAUNCHED
setsid nohup bash $B/xfer/queue_after_l8.sh > $B/logs/queue_after_l8.log 2>&1 < /dev/null &
