#!/bin/bash
B=/home/shensicheng/qyd/pans
for T in q14 q32; do
  cd $B/src_$T && . $B/env_timing.sh && . $B/env_$T.sh && export PANS_ROOT=$B/src_$T
  $B/venv/bin/python scripts/v29m_timing.py a100_timing_$T --tag s800 > $B/logs/a100_timing_$T.log 2>&1
done
echo ALL_TIMING_DONE
