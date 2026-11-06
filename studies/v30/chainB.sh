#!/bin/bash
# v30 phase B (after phase A): selections, remaining quality groups, report, then SSD timing.
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v30/logs
E1=$($P -c "print(repr(100*16/28/4))"); E2=$($P -c "print(repr(100*20/28/4))"); E3=$($P -c "print(repr(100*24/28/4))")
waitfor() { until grep -q "$2" "$1" 2>/dev/null; do sleep 60; done; }
(
  waitfor $L/kvscore.log KVSCORE_ALL_DONE
  for t in sst2 subj trec rte; do for m in cake lava; do
    for b in 10 $E1 $E2 $E3 25 50; do $P scripts/v30_select_build.py static $m $t $b >> $L/select_build.log 2>&1; done
    for b in $E1 25; do $P scripts/v30_select_build.py profile $m $t $b >> $L/select_build.log 2>&1; done
    $P scripts/v30_select_build.py oracle $m $t 25 test >> $L/select_build.log 2>&1
  done; done; echo SELECT_BUILD_DONE >> $L/select_build.log
  $P scripts/v30_quality.py cake_static,cake_budget,cake_oracle,lava_oracle 3 > $L/q_gpu3.log 2>&1
) &
(
  waitfor $L/p4_readers.log P4_ALL_DONE
  $P scripts/v30_quality.py ns_p4_dev,p4_dev 2 > $L/q_gpu2_a.log 2>&1
  $P scripts/v30_report.py select-only > $L/p4_select.log 2>&1
  waitfor $L/select_build.log SELECT_BUILD_DONE
  $P scripts/v30_quality.py p4_test,lava_budget 2 > $L/q_gpu2_b.log 2>&1
) &
(
  waitfor $L/q_gpu1_a.log V30_QUALITY_GROUPS_DONE
  waitfor $L/select_build.log SELECT_BUILD_DONE
  $P scripts/v30_quality.py lava_static 1 > $L/q_gpu1_b.log 2>&1
) &
wait
$P scripts/v30_report.py > $L/report_quality.log 2>&1
echo QUALITY_PHASE_DONE >> $L/chainB.log
$P scripts/v30_timing.py > $L/timing.log 2>&1
$P scripts/v30_report.py > $L/report_final.log 2>&1
echo CHAIN_B_DONE >> $L/chainB.log
