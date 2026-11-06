#!/bin/bash
# Llama-3.1-8B: resume after the 28-layer geometry bug (teacher captures are reused). Readers -> dev -> P4 selection -> test.
set -u
B=/data1/pans_l8; S=$B/raps_l8; M=/data1/llm/meta/Meta-Llama-3.1-8B-Instruct; L=$B/logs
mkdir $B/lock_pipeline2 2>/dev/null || exit 0
export V30_ROOT=$S V30_MODEL=$M V30_PCACHE=$B/work_l8/pcache_c16 V30_STEPS=800 PANS_MODEL_PATH=$M PANS_PY=/home/panzihang/venvs/vllm-stable/bin/python \
       PANS_PROBE_QUERY_HEADS=0,4,8,12,16,20,24,28 PANS_SELECTOR_KV_HEAD_IDS=0,1,2,3,4,5,6,7 TMPDIR=$B/tmp
ulimit -n 1048576
P=$PANS_PY; cd $S
cp /home/panzihang/src/prism_layerkv/scripts/v30_p4_readers.py scripts/
step() { echo "== $1 $(date)" >> $L/pipeline2.log; shift; "$@" >> $L/pipeline2.log 2>&1 || { echo "STEP_FAILED $*" >> $L/pipeline2.log; }; }
( $P scripts/v30_p4_readers.py sst2 0 32-39 --max-feasible >> $L/p4_readers2.log 2>&1 ) &
( $P scripts/v30_p4_readers.py subj 1 40-47 --max-feasible >> $L/p4_readers2.log 2>&1 ) &
( $P scripts/v30_p4_readers.py trec 2 48-55 --max-feasible >> $L/p4_readers2.log 2>&1 ) &
( $P scripts/v30_p4_readers.py rte 3 56-59,12-15 --max-feasible >> $L/p4_readers2.log 2>&1 ) &
wait; echo "== readers done $(date)" >> $L/pipeline2.log; grep -c P4_READERS_DONE $L/p4_readers2.log >> $L/pipeline2.log
step dev $P scripts/v30_quality.py ns_p4_dev,p4_dev 0,1,2,3
step select $P scripts/v30_report.py select-only
step test $P scripts/v30_quality.py ns_p4_test,p4_test,ns_eq 0,1,2,3
echo L8_PIPELINE2_DONE >> $L/pipeline2.log
