#!/bin/bash
B=/home/shensicheng/qyd/pans
ps -u shensicheng -o pid,args | awk '/queue_timing_sys.sh/ && !/awk/ {print $1}' | xargs -r kill
sleep 1
ps -u shensicheng -o pid,args | awk '/v29m_timing_sys.py/ && !/awk/ {print $1}' | xargs -r kill
sleep 3
ps -u shensicheng -o pid,args | awk '/run_evidence_v26|v29_baseline_entry/ && /a100_sys_q14/ && !/awk/ {print $1}' | xargs -r kill
sleep 2
mv $B/src_q14/studies/v29/a100_sys_q14 $B/src_q14/studies/v29/_a100_sys_q14_wrongmodel_failed
mv $B/logs/a100_sys_q14.log $B/logs/a100_sys_q14.wrongmodel.log
setsid nohup bash $B/xfer/queue_timing_sys2.sh > $B/logs/queue_timing_sys2.log 2>&1 < /dev/null &
sleep 15; cat $B/logs/queue_timing_sys2.log; head -1 $B/logs/a100_sys_q14.log
