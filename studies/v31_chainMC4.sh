#!/bin/bash
# v31: single-instance (lock dir) clean rebuild with the 4-slot store, then the dev screen
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v31/logs
mkdir studies/v31/lock_mc4 2>/dev/null || exit 0
for p in $(pgrep -u panzihang -f "v31_chainMC3.sh|v31_chainMC2.sh|scripts/v31_mc_screen.py|scripts/build_selector_index.py"); do kill $p; done; sleep 8
A=studies/v31/_failed_concurrent_rebuilds; mkdir -p $A
for x in runtime_data/v31_mc_store runtime_data/v31_mc_pcache assets/v31_mc_index studies/v31/mc_screen; do [ -e $x ] && mv $x $A/$(basename $x)_$(date +%s); done
echo "MC4 start $(date)" >> $L/chainMC.log
$P scripts/v31_mc_screen.py build 1 > $L/build4.log 2>&1 || { echo BUILD4_FAILED >> $L/chainMC.log; exit 1; }
$P scripts/v31_mc_screen.py screen 1,2,3 > $L/screen4.log 2>&1 || { echo SCREEN4_FAILED >> $L/chainMC.log; exit 1; }
echo CHAIN_MC4_DONE >> $L/chainMC.log
