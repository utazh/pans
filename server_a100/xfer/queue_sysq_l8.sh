#!/bin/bash
# Llama 计时结束（磁盘可用）后：ContiguousKV / IMPRESS 在 25% 完整测试集上的准确率。
B=/home/shensicheng/qyd/pans; Q=$B/xfer/sysquality.sh; L=$B/logs
until grep -q L8_TIMING_DONE $L/a100_sys_l8.log; do sleep 60; done
setsid nohup bash $Q l8 0 0-7,128-135 impress sst2 > $L/sysq_l8_imp_g0.log 2>&1 < /dev/null &
setsid nohup bash $Q l8 1 8-15,136-143 impress subj > $L/sysq_l8_imp_g1.log 2>&1 < /dev/null &
setsid nohup bash $Q l8 2 16-23,144-151 impress trec rte > $L/sysq_l8_imp_g2.log 2>&1 < /dev/null &
setsid nohup bash $Q l8 4 84-91,212-219 contigkv sst2 subj trec rte > $L/sysq_l8_ckv_g4.log 2>&1 < /dev/null &
echo SYSQ_L8_LAUNCHED
