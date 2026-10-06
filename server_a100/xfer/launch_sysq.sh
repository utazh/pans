#!/bin/bash
B=/home/shensicheng/qyd/pans; Q=$B/xfer/sysquality.sh; L=$B/logs
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
setsid nohup bash $Q q32 0 0-7,128-135 impress subj > $L/sysq_q32_imp_g0.log 2>&1 < /dev/null &
setsid nohup bash $Q q32 1 8-15,136-143 impress sst2 > $L/sysq_q32_imp_g1.log 2>&1 < /dev/null &
setsid nohup bash $Q q32 2 16-23,144-151 impress trec rte > $L/sysq_q32_imp_g2.log 2>&1 < /dev/null &
setsid nohup bash $Q q14 3 48-55,176-183 impress subj rte > $L/sysq_q14_imp_g3.log 2>&1 < /dev/null &
setsid nohup bash $Q q14 5 64-71,192-199 impress sst2 trec > $L/sysq_q14_imp_g5.log 2>&1 < /dev/null &
setsid nohup bash -c "bash $Q q14 6 84-91,212-219 contigkv sst2 subj trec rte; bash $Q q32 6 84-91,212-219 contigkv sst2 subj trec rte" > $L/sysq_ckv_g6.log 2>&1 < /dev/null &
sleep 2; echo LAUNCHED6
