#!/bin/bash
# v33: AS setup (runner, build, test quality) -> same-block cold+warm timing of all 8 methods (7B, 4 tasks, 25 %)
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v33/logs; mkdir -p $L
mkdir studies/v33/lock_chain 2>/dev/null || exit 0
$P scripts/v33_as_setup.py runner > $L/as_runner.log 2>&1 || { echo RUNNER_FAILED >> $L/chain.log; exit 1; }
$P scripts/v33_as_setup.py build 0 > $L/as_build.log 2>&1 || { echo BUILD_FAILED >> $L/chain.log; exit 1; }
$P scripts/v33_as_setup.py quality 0,1,3 > $L/as_quality.log 2>&1 || { echo ASQ_FAILED >> $L/chain.log; exit 1; }
echo AS_READY >> $L/chain.log
$P scripts/v33_timing.py > $L/timing.log 2>&1 || { echo TIMING_FAILED >> $L/chain.log; exit 1; }
echo V33_DONE >> $L/chain.log
