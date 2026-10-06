#!/bin/bash
B=/home/shensicheng/qyd/pans
ps -u shensicheng -o pid,args | grep "run_evidence_v26" | grep "a100_timing_q7" | grep -v grep | awk '{print $1}' | xargs -r kill
sleep 2
mv $B/src/studies/v29/a100_timing_q7 $B/src/studies/v29/_a100_timing_q7_gpu6_partial29
mv $B/logs/a100_timing_q7.log $B/logs/a100_timing_q7.gpu6_partial.log
sed -i "s/PANS_TIMING_GPU=6/PANS_TIMING_GPU=4/; s/PANS_WATCH_GPUS=6/PANS_WATCH_GPUS=4/" $B/env_timing.sh
grep GPU $B/env_timing.sh
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n 5p
cd $B/src && . $B/env_timing.sh && setsid nohup $B/venv/bin/python scripts/v29a_timing.py a100_timing_q7 > $B/logs/a100_timing_q7.log 2>&1 < /dev/null &
sleep 3; echo RESTARTED_ON_GPU4
