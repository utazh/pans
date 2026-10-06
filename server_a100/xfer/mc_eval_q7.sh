#!/bin/bash
B=/home/shensicheng/qyd/pans
until grep -q -E "MINICACHE_BUILD_DONE|Traceback" $B/logs/minicache_build_q7.log; do sleep 20; done
grep -q MINICACHE_BUILD_DONE $B/logs/minicache_build_q7.log || { echo BUILD_FAILED; exit 1; }
export PANS_EVAL_STORE=$B/work/q7/mc_store PANS_EVAL_PCACHE=$B/work/q7/mc_pcache
for t in trec sst2 rte subj; do bash $B/xfer/q_eval.sh 6 $t 25 q7_${t}_b25_minicache > $B/logs/q_eval_q7_${t}_b25_mc.log 2>&1; done
echo MC_EVAL_ALL_DONE
