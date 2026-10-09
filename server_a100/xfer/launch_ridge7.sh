#!/bin/bash
# G1 ridge-only（7B，25%）：读取器 = 每个前缀上的闭式带正则最小二乘（训练 0 步），其余流程与正式配方相同
# （同一几何共享图 g16 → S3/S2/S1 → 500 条开发集按冻结规则选档 → 完整测试集）。用于回答“梯度训练是否必要”。
B=/home/shensicheng/qyd/pans; cd $B/src
export PANS_MODEL_PATH=$B/models/Qwen2.5-7B-Instruct PANS_PY=$B/venv/bin/python PANS_PROBE_QUERY_HEADS=0,7,14,21 PANS_SELECTOR_KV_HEAD_IDS=0,1,2,3 TMPDIR=$B/tmp
ulimit -n 1048576
CELLS=(sst2_b25 subj_b25 trec_b25 rte_b25); GPUS=(4 4 6 6)   # GPU 0-3 有他人任务，用空闲卡
for i in 0 1 2 3; do
  c=${CELLS[$i]}; lo=$((64 + 8 * i)); hi=$((lo + 7))
  setsid nohup $PANS_PY scripts/v29m_pipeline.py all --cells $c --gpu ${GPUS[$i]} --cores $lo-$hi --steps 0 --tag ridge \
     > $B/logs/ridge7_$c.log 2>&1 < /dev/null &
  sleep 20
done
echo RIDGE7_LAUNCHED
