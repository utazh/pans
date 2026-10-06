#!/bin/bash
B=/home/shensicheng/qyd/pans
cd $B/src && git -c user.email=a100@local -c user.name=a100 commit -qam "v29m: geometry rule requires identical selections (v26 compatibility)"
cd $B/src_q14 && git merge -q a100 && git log --oneline -1
for c in rte_b25 subj_b25; do
  for p in $(pgrep -u shensicheng -f "scripts/v29m_pipeline.py all --cells $c "); do pkill -TERM -P $p; kill $p; done
done
sleep 5
. $B/env_q14.sh; ulimit -n 1048576
for spec in "sst2_b25 1 0-7,128-135" "rte_b25 2 8-15,136-143" "subj_b25 3 48-55,176-183"; do set -- $spec
  mv $B/logs/v29m_q14_s800_$1.log $B/logs/v29m_q14_s800_$1.prev.log 2>/dev/null
  setsid nohup $B/venv/bin/python scripts/v29m_pipeline.py all --cells $1 --gpu $2 --cores $3 --steps 800 --tag s800 --reuse-from v29m > $B/logs/v29m_q14_s800_$1.log 2>&1 < /dev/null &
done
sleep 3; echo RESTARTED
