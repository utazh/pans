#!/bin/bash
# Timing queue on GPU 2 (single instance): Llama SSD Pcache copies -> 7B flagged-block reruns (SST-2 cold x6, SUBJ cold b0) ->
# Llama 8-method SSD-cold + warm timing. Nothing of ours may use GPU 3 meanwhile (gate watches GPUs 2,3).
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v34/logs; mkdir -p $L
mkdir studies/v34/lock_timing 2>/dev/null || exit 0
S=runtime_data; W=/data1/pans_l8/work_l8
for x in c16:pcache_c16 imp:pcache_impress_c64 as:pcache_as_c64; do d=$S/v34_l8_${x%%:*}_ssd; [ -d $d ] || cp -a $W/${x#*:} $d; done
echo "COPIES_DONE $(date)" >> $L/chain.log
V33_STUDY=../v33/timing_fix $P scripts/v33_timing.py --only sst2_b25:preserve:0,sst2_b25:preserve:1,sst2_b25:preserve:2,sst2_b25:preserve:3,sst2_b25:preserve:4,sst2_b25:preserve:5,subj_b25:preserve:0 > $L/fix7b.log 2>&1 \
  && echo "FIX7B_DONE $(date)" >> $L/chain.log || echo FIX7B_FAILED >> $L/chain.log
M=/data1/llm/meta/Meta-Llama-3.1-8B-Instruct; R8=/data1/pans_l8/raps_l8
cp scripts/v33_timing.py scripts/v30_common.py scripts/v30_round4_ext.py scripts/v30_entry.py scripts/v33_as_entry.py $R8/scripts/
( cd $R8 && export V30_ROOT=$R8 V30_MODEL=$M PANS_ROOT=$R8 PANS_MODEL_PATH=$M PANS_PY=$P PANS_PROBE_QUERY_HEADS=0,4,8,12,16,20,24,28 \
    PANS_SELECTOR_KV_HEAD_IDS=0,1,2,3,4,5,6,7 PANS_TIMING_GPU=2 PANS_TIMING_CORES=28-31,60-63 PANS_TIMING_DISK=sda PANS_WATCH_GPUS=2,3 \
    PANS_BACKGROUND_CPUS=0-11,16-27 PANS_NUMA_CPUS=16-31,48-63 TMPDIR=/data1/pans_l8/tmp \
    V33_C16=/home/panzihang/src/prism_layerkv/$S/v34_l8_c16_ssd V33_C64=/home/panzihang/src/prism_layerkv/$S/v34_l8_imp_ssd \
    V33_CAS=/home/panzihang/src/prism_layerkv/$S/v34_l8_as_ssd V33_REORDER=$W/impress_reorder_fit4.json V33_STUDY=../v34/l8_timing && \
  ulimit -n 1048576 && $P scripts/v33_timing.py > /home/panzihang/src/prism_layerkv/$L/l8_timing.log 2>&1 ) \
  && echo "L8_TIMING_DONE $(date)" >> $L/chain.log || echo L8_TIMING_FAILED >> $L/chain.log
