#!/bin/bash
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v30/logs
mkdir -p configs/v30_sel/_superseded_mean_mismatch
for t in sst2 subj trec rte; do for m in cake lava; do
  mv configs/v30_sel/${m}budget_${t}_k14p2857.json configs/v30_sel/_superseded_mean_mismatch/
  $P scripts/v30_select_build.py profile $m $t 14.2857 >> $L/select_build_fix.log 2>&1
done; done
$P scripts/v30_quality.py cake_budget,lava_budget 1,2,3 > $L/q_budget_fix.log 2>&1
$P scripts/v30_report.py > $L/report_final2.log 2>&1
echo FIX_DONE >> $L/q_budget_fix.log
