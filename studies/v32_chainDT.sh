#!/bin/bash
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v32/logs; mkdir -p $L
mkdir studies/v32/lock_dt 2>/dev/null || exit 0
for s in train dev test; do $P scripts/v32_direct.py $s 0,1,2,3 > $L/dt_$s.log 2>&1 || { echo DT_${s}_FAILED >> $L/chain.log; exit 1; }; done
$P scripts/v32_direct.py report > $L/dt_report.log 2>&1; echo DT_DONE >> $L/chain.log
