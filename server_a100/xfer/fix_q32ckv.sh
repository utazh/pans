#!/bin/bash
B=/home/shensicheng/qyd/pans
ps -u shensicheng -o pid,args | awk '/queue_timing_sys.sh/ && !/awk/ {print $1}' | xargs -r kill
sleep 1
ps -u shensicheng -o pid,args | awk '/v29m_timing_sys.py/ && !/awk/ {print $1}' | xargs -r kill
sleep 2
ls -d $B/src/studies/v29/a100_sys_q7 2>/dev/null && mv $B/src/studies/v29/a100_sys_q7 $B/src/studies/v29/_a100_sys_q7_preempted_$(date +%s)
Q=$B/xfer/sysquality.sh
setsid nohup bash $Q q32 1 8-15,136-143 contigkv sst2 subj > $B/logs/sysq_q32_ckv_g1b.log 2>&1 < /dev/null &
setsid nohup bash $Q q32 5 64-71,192-199 contigkv trec rte > $B/logs/sysq_q32_ckv_g5b.log 2>&1 < /dev/null &
sleep 5
setsid nohup bash $B/xfer/queue_timing_sys.sh > $B/logs/queue_timing_sys.log 2>&1 < /dev/null &
sleep 2; ps -u shensicheng -o pid,args | awk '/sysquality|queue_timing/ && !/awk/' | cut -c1-110
