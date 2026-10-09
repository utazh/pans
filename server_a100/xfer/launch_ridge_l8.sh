#!/bin/bash
# G1 ridge-only（Llama-8B，25%）：与 v29m_s800（800 步训练）同一流水线、同一共享图、同一 teacher 采集与 I 评估（--reuse-from），
# 只把训练步数设为 0（读取器 = 闭式带正则最小二乘初值）。
B=/home/shensicheng/qyd/pans; source $B/env_l8.sh; ulimit -n 1048576; cd $B/src_l8
CELLS=(sst2_b25 subj_b25 trec_b25 rte_b25); GPUS=(4 4 6 6)
for i in 0 1 2 3; do
  c=${CELLS[$i]}; lo=$((96 + 4 * i)); hi=$((lo + 3))
  setsid nohup $PANS_PY scripts/v29m_pipeline.py all --cells $c --gpu ${GPUS[$i]} --cores $lo-$hi --steps 0 --tag ridge \
     --reuse-from v29m_s800 > $B/logs/ridge_l8_$c.log 2>&1 < /dev/null &
  sleep 20
done
echo RIDGE_L8_LAUNCHED
