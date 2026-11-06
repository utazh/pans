#!/bin/bash
# v30 phase A: GPU2 P4 readers; GPU3 CAKE/LAVa statistics (prefix windows, then oracle on the test sets)
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v30/logs; mkdir -p $L
( for t in sst2 trec rte; do $P scripts/v30_p4_readers.py $t 2 48-55 >> $L/p4_readers.log 2>&1 || echo "FAIL $t" >> $L/p4_readers.log; done; echo P4_ALL_DONE >> $L/p4_readers.log ) &
( for t in sst2 subj rte; do CUDA_VISIBLE_DEVICES=3 taskset -c 56-59,12-15 $P scripts/v30_kvscore.py prefix $t >> $L/kvscore.log 2>&1; done
  for t in sst2 subj trec rte; do CUDA_VISIBLE_DEVICES=3 taskset -c 56-59,12-15 $P scripts/v30_kvscore.py oracle $t studies/v28_frozen/bundles/${t}_b25/eval >> $L/kvscore.log 2>&1; done
  echo KVSCORE_ALL_DONE >> $L/kvscore.log ) &
wait
