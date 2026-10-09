#!/bin/bash
# Llama 模型级读取器（留一法）两条链；每个目标在其余 3 个任务上训练 800 步。
B=/home/shensicheng/qyd/pans; R=$B/xfer/mlt_run.sh
( bash $R src_l8 env_l8.sh s800 subj_b25 sst2_b25,trec_b25,rte_b25 800 mlt > $B/logs/mlt_l8_subj.log 2>&1
  bash $R src_l8 env_l8.sh s800 rte_b25 sst2_b25,subj_b25,trec_b25 800 mlt > $B/logs/mlt_l8_rte.log 2>&1 ) &
sleep 600
( bash $R src_l8 env_l8.sh s800 trec_b25 sst2_b25,subj_b25,rte_b25 800 mlt > $B/logs/mlt_l8_trec.log 2>&1
  bash $R src_l8 env_l8.sh s800 sst2_b25 subj_b25,trec_b25,rte_b25 800 mlt > $B/logs/mlt_l8_sst2.log 2>&1 ) &
wait; echo MLT_L8_ALL_DONE
