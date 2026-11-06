#!/bin/bash
# After the Llama timing: Llama flagged-block rerun (SUBJ cold b4) -> QuALITY baseline Pcaches to SSD -> QuALITY 8-method cold+warm timing.
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v34/logs
mkdir studies/v34/lock_timing2 2>/dev/null || exit 0
until grep -qE "L8_TIMING_DONE|L8_TIMING_FAILED" $L/chain.log; do sleep 120; done
M=/data1/llm/meta/Meta-Llama-3.1-8B-Instruct; R8=/data1/pans_l8/raps_l8; W=/data1/pans_l8/work_l8; S=runtime_data
( cd $R8 && export V30_ROOT=$R8 V30_MODEL=$M PANS_ROOT=$R8 PANS_MODEL_PATH=$M PANS_PY=$P PANS_PROBE_QUERY_HEADS=0,4,8,12,16,20,24,28 \
    PANS_SELECTOR_KV_HEAD_IDS=0,1,2,3,4,5,6,7 PANS_TIMING_GPU=2 PANS_TIMING_CORES=28-31,60-63 PANS_TIMING_DISK=sda PANS_WATCH_GPUS=2,3 \
    PANS_BACKGROUND_CPUS=0-11,16-27 PANS_NUMA_CPUS=16-31,48-63 TMPDIR=/data1/pans_l8/tmp \
    V33_C16=/home/panzihang/src/prism_layerkv/$S/v34_l8_c16_ssd V33_C64=/home/panzihang/src/prism_layerkv/$S/v34_l8_imp_ssd \
    V33_CAS=/home/panzihang/src/prism_layerkv/$S/v34_l8_as_ssd V33_REORDER=$W/impress_reorder_fit4.json V33_STUDY=../v34/l8_timing_fix && \
  ulimit -n 1048576 && $P scripts/v33_timing.py --only subj_b25:preserve:4 > /home/panzihang/src/prism_layerkv/$L/l8_fix.log 2>&1 ) \
  && echo "L8_FIX_DONE $(date)" >> $L/chain.log || echo L8_FIX_FAILED >> $L/chain.log
for s in f0 f1 f2; do
  [ -d $S/v35_qa_imp_c64_${s}_ssd ] || cp -a /data1/pans_qa/imp_c64_$s $S/v35_qa_imp_c64_${s}_ssd
  [ -d $S/v35_qa_as_c64_${s}_ssd ] || cp -a /data1/pans_qa/as_c64_$s $S/v35_qa_as_c64_${s}_ssd
done
echo "QA_COPIES_DONE $(date)" >> $L/chain.log
ulimit -n 1048576
$P scripts/v35_timing_qa.py > $L/qa_timing.log 2>&1 && echo "QA_TIMING_DONE $(date)" >> $L/chain.log || echo QA_TIMING_FAILED >> $L/chain.log
