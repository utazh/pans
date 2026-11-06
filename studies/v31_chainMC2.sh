#!/bin/bash
# v31 rebuild with the 4-slot store, then the dev screen (previous 3-slot store/pcache/index and failed runs archived)
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v31/logs
A=studies/v31/_failed_3slot_store; mkdir -p $A
mv runtime_data/v31_mc_store $A/store; mv runtime_data/v31_mc_pcache $A/pcache; mv assets/v31_mc_index $A/index; mv studies/v31/mc_screen $A/mc_screen
$P scripts/v31_mc_screen.py build 1 > $L/build2.log 2>&1 || { echo BUILD2_FAILED >> $L/chainMC.log; exit 1; }
$P scripts/v31_mc_screen.py screen 1,2,3 > $L/screen2.log 2>&1 || { echo SCREEN2_FAILED >> $L/chainMC.log; exit 1; }
echo CHAIN_MC2_DONE >> $L/chainMC.log
