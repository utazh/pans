#!/bin/bash
# Llama-3.1-8B on the 3090, after l8build.sh: our system (NS P8/P4, RAPS-P4 readers 800 steps, frozen dev rule), CAKE/LAVa,
# ContiguousKV / IMPRESS / AS+H2O+LRU quality. GPUs 0,1 only (GPU 2 = 7B timing, GPU 3 must stay idle for its gate);
# all Llama data on /data1 (HDD) so the SSD timing gate is not disturbed. Timing comes after the 7B campaign.
set -u
B=/data1/pans_l8; S=$B/raps_l8; D=$B/work_l8; M=/data1/llm/meta/Meta-Llama-3.1-8B-Instruct; L=$B/logs
mkdir $B/lock_pipeline 2>/dev/null || exit 0
until grep -qE "L8_BUILD_DONE" $L/build.log 2>/dev/null; do sleep 60; done
export V30_ROOT=$S V30_MODEL=$M V30_PCACHE=$D/pcache_c16 V30_STEPS=800 PANS_MODEL_PATH=$M PANS_PY=/home/panzihang/venvs/vllm-stable/bin/python \
       PANS_PROBE_QUERY_HEADS=0,4,8,12,16,20,24,28 PANS_SELECTOR_KV_HEAD_IDS=0,1,2,3,4,5,6,7 TMPDIR=$B/tmp
ulimit -n 1048576
P=$PANS_PY; cd $S
cp /home/panzihang/src/prism_layerkv/scripts/v30_*.py /home/panzihang/src/prism_layerkv/scripts/v33_as_entry.py /home/panzihang/src/prism_layerkv/scripts/v34_*.py scripts/
[ -f scripts/run_round4_methods_as.py ] || cp scripts/run_round4_methods.py scripts/run_round4_methods_as.py
step() { echo "== $1 $(date)" >> $L/pipeline.log; shift; "$@" >> $L/pipeline.log 2>&1 || { echo "STEP_FAILED $*" >> $L/pipeline.log; }; }
step sanity_ns_p8 $P scripts/v30_quality.py ns_p8_test 0,1
( for t in sst2 trec; do $P scripts/v30_p4_readers.py $t 0 32-39 --max-feasible >> $L/p4_readers.log 2>&1; done ) &
( for t in subj rte; do $P scripts/v30_p4_readers.py $t 1 40-47 --max-feasible >> $L/p4_readers.log 2>&1; done ) &
wait; echo "== readers done $(date)" >> $L/pipeline.log
step dev $P scripts/v30_quality.py ns_p4_dev,p4_dev 0,1
step select $P scripts/v30_report.py select-only
step test $P scripts/v30_quality.py ns_p4_test,p4_test,ns_eq 0,1
E1=$($P -c "print(repr(100*16/28/4))"); E2=$($P -c "print(repr(100*20/28/4))"); E3=$($P -c "print(repr(100*24/28/4))")
for t in sst2 subj trec rte; do CUDA_VISIBLE_DEVICES=0 $P scripts/v30_kvscore.py prefix $t >> $L/kvscore.log 2>&1; done
for t in sst2 subj trec rte; do for m in cake lava; do for b in 10 $E1 $E2 $E3 25 50; do $P scripts/v30_select_build.py static $m $t $b >> $L/select.log 2>&1; done; done; done
step cakelava $P scripts/v30_quality.py cake_static,lava_static 0,1
# IMPRESS c64 (A100 manifest reused; same steps as xfer/build_impress.sh) and AS c64 plain Pcaches, then baseline quality
MAN=$D/impress_reorder_fit4.json; C64=$D/pcache_impress_c64; CAS=$D/pcache_as_c64
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTHONPATH=src:runtime_deps_v24
if [ ! -f $C64/.contiguous_fuxian_complete ]; then
  $P scripts/run_round4_methods.py --method impress --build-kv --kv-dir $C64 --impress-reorder $MAN --task sst2 --budget 5 --samples 2 --warmup 0 \
     --bundle-dir studies/v28_frozen/bundles/sst2_b5/eval --output $S/studies/v34_impress_build >> $L/impress_build.log 2>&1
  $P /home/panzihang/src/pans/xfer/fill_impress_head.py $S $C64 $MAN --write >> $L/impress_build.log 2>&1
  $P - "$C64" "$(sha256sum $MAN | cut -d' ' -f1)" >> $L/impress_build.log 2>&1 <<'PYEOF'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1])
for f in p.iterdir():
    if f.is_file():
        with f.open('rb') as h: os.fsync(h.fileno())
(p / '.contiguous_fuxian_complete').write_text(json.dumps(dict(schema_version=2, method='impress', chunk_size=64, selector_kv_head_ids=[0],
    online_selection=True, impress_reorder_sha256=sys.argv[2], built_by='scripts/run_round4_methods.py --build-kv (3090 v34)',
    prefix_store='runtime_data/prefix_store'), indent=2))
print('marker written')
PYEOF
fi
[ -f $CAS/.contiguous_fuxian_complete ] || [ -d $CAS ] || $P scripts/run_round4_methods_as.py --method as_h2o_lru --build-kv --kv-dir $CAS --task sst2 --budget 25 \
     --samples 2 --warmup 0 --bundle-dir studies/v28_frozen/bundles/sst2_b25/eval --output $S/studies/v34_as_build >> $L/as_build.log 2>&1
unset CUDA_VISIBLE_DEVICES
step baselines $P scripts/v34_baseline_quality.py 0,1 $C64 $MAN $CAS
echo L8_PIPELINE_DONE >> $L/pipeline.log
