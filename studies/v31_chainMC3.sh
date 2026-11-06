#!/bin/bash
# v31: stop the mistaken 3-slot rebuild, archive its partial outputs, rebuild with the 4-slot store, then screen
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v31/logs
for p in $(pgrep -u panzihang -f "v31_chainMC2.sh|scripts/v31_mc_screen.py|scripts/build_selector_index.py"); do kill $p; done; sleep 5
A=studies/v31/_failed_3slot_store2; mkdir -p $A
for x in runtime_data/v31_mc_store runtime_data/v31_mc_pcache assets/v31_mc_index studies/v31/mc_screen; do [ -e $x ] && mv $x $A/$(basename $x); done
grep -q STORE_SLOTS scripts/v31_mc_screen.py || { echo PATCH_MISSING >> $L/chainMC.log; exit 1; }
$P scripts/v31_mc_screen.py build 1 > $L/build3.log 2>&1 || { echo BUILD3_FAILED >> $L/chainMC.log; exit 1; }
$P scripts/v31_mc_screen.py screen 1,2,3 > $L/screen3.log 2>&1 || { echo SCREEN3_FAILED >> $L/chainMC.log; exit 1; }
echo CHAIN_MC3_DONE >> $L/chainMC.log
