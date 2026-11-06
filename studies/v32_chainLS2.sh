#!/bin/bash
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v32/logs; mkdir -p $L
mkdir studies/v32/lock_ls2 2>/dev/null || exit 0
[ -d data/v32/sens64 ] || PYTHONPATH=src $P scripts/prepare_v32_sens64.py > $L/sens64.log 2>&1 || { echo SENS64_FAILED >> $L/chain.log; exit 1; }
[ -d studies/v32/layer_sens ] && mv studies/v32/layer_sens studies/v32/_failed_layer_sens_fitleak
for s in fit dev test; do $P scripts/v32_layer_sens.py $s 0,1,2,3 > $L/ls2_$s.log 2>&1 || { echo LS2_${s}_FAILED >> $L/chain.log; exit 1; }; done
$P scripts/v32_layer_sens.py report > $L/ls2_report.log 2>&1; echo LS2_DONE >> $L/chain.log
