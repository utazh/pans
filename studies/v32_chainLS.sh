#!/bin/bash
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v32/logs; mkdir -p $L
mkdir studies/v32/lock_ls 2>/dev/null || exit 0
for s in fit dev test; do $P scripts/v32_layer_sens.py $s 1,2,3 > $L/ls_$s.log 2>&1 || { echo LS_${s}_FAILED >> $L/chain.log; exit 1; }; done
$P scripts/v32_layer_sens.py report > $L/ls_report.log 2>&1; echo LS_DONE >> $L/chain.log
