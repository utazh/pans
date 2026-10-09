#!/bin/bash
# 等 Llama 基础产物就绪 → 4 个 25% 单元并行跑完整 RAPS 流程（capture→plan→train 800 步→derive→dev 选档→test）。
# GPU 0,1,2,4 各一个单元；核 64-95 每单元 8 个（与并发实验的 100-123 不重叠）。
B=/home/shensicheng/qyd/pans
until grep -q MODEL_READY_l8 $B/logs/setup_l8.log; do
  if ! pgrep -u shensicheng -f "[s]etup_l8.sh" > /dev/null; then echo "SETUP_L8_DIED"; tail -20 $B/logs/setup_l8.log; exit 1; fi
  sleep 60
done
source $B/env_l8.sh; ulimit -n 1048576; cd $B/src_l8
CELLS=(sst2_b25 subj_b25 trec_b25 rte_b25); GPUS=(0 1 2 4)
for i in 0 1 2 3; do
  c=${CELLS[$i]}; lo=$((64 + 8 * i)); hi=$((lo + 7))
  setsid nohup $PANS_PY scripts/v29m_pipeline.py all --cells $c --gpu ${GPUS[$i]} --cores $lo-$hi --steps 800 --tag s800 \
     > $B/logs/l8_pipe_$c.log 2>&1 < /dev/null &
  sleep 30
done
echo L8_PIPELINES_LAUNCHED
