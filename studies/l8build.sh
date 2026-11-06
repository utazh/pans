#!/bin/bash
# Llama-3.1-8B base artifacts on the 3090 (same commands as the A100 xfer/build_base.sh, paths moved): prefix store ->
# chunk-16 Pcache (+ completion marker, v29 schema) -> K4 selector index. Starts after the v33 AS quality stage (GPU 0).
set -euo pipefail
B=/data1/pans_l8; S=$B/raps_l8; D=$B/work_l8; M=/data1/llm/meta/Meta-Llama-3.1-8B-Instruct; L=$B/logs
PQH=0,4,8,12,16,20,24,28; SKV=0,1,2,3,4,5,6,7
mkdir $B/lock_build 2>/dev/null || exit 0
until grep -qE "AS_READY|FAILED" /home/panzihang/src/prism_layerkv/studies/v33/logs/chain.log 2>/dev/null; do sleep 60; done
ulimit -n 1048576
export TMPDIR=$B/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTHONPATH=$S/src CUDA_VISIBLE_DEVICES=0; mkdir -p $TMPDIR
PY=/home/panzihang/venvs/vllm-stable/bin/python; cd $S
mv $D/selector_index_k4_g32 $D/_a100_selector_index_k4_g32   # rebuilt below from the 3090 Pcache (kept for comparison)
echo "== prefix store $(date)" >> $L/build.log
$PY -m contiguous_fuxian.sparse_qwen_reprefill prepare --model-path $M --bundle-dir $D/bundles_strict --store-root $D/prefix_store \
   --tasks sst2,subj,trec,rte --samples-per-task 1 --dtype bfloat16 --device cuda >> $L/build.log 2>&1
echo "== pcache $(date)" >> $L/build.log
$PY -m contiguous_fuxian.flexgen_qwen_reprefill --model-path $M --bundle-dir $D/bundles_strict --store-root $D/prefix_store \
   --plan configs/qwen25_k005_contigkv.json --output-dir $D/build_pcache_run_3090 --flexgen-root vendor/flexgen --flexgen-kv-dir $D/pcache_c16 \
   --tasks sst2 --store-tasks sst2,subj,trec,rte --samples-per-task 1 --max-tokens 1 --dtype bfloat16 --device cuda --gpu-cache-mb 0 \
   --cpu-cache-mb 0 --cache-type CKLFU --online-selection --probe-query-heads $PQH --selector-kv-head-ids $SKV --period-size 8 \
   --subperiod-size 4 --expected-keep-ratio 0.05 >> $L/build.log 2>&1
$PY - "$D/pcache_c16" "$SKV" >> $L/build.log 2>&1 <<'PYEOF'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1]); heads = [int(h) for h in sys.argv[2].split(',')]
for f in p.iterdir():
    if f.is_file():
        with f.open('rb') as h: os.fsync(h.fileno())
(p / '.contiguous_fuxian_complete').write_text(json.dumps(dict(schema_version=3, method='a100_rebuild_prefix_native_full_kv',
    chunk_size=16, selector_kv_head_ids=heads, online_selection=True, physical_layout='plain-logical-token-order',
    impress_reorder_sha256=None, registered_store_tasks=['sst2', 'subj', 'trec', 'rte'], build_report='build_pcache_run_3090'), indent=2))
print('marker written', heads)
PYEOF
echo "== index $(date)" >> $L/build.log
$PY scripts/build_selector_index.py --source-pcache-dir $D/pcache_c16 --store-root $D/prefix_store --output-dir $D/selector_index_k4_g32 \
   --model-path $M --tasks sst2,subj,trec,rte --source-task-order sst2,subj,trec,rte --selector-kv-head-ids $SKV --group-size 32 >> $L/build.log 2>&1
du -sh $D/* >> $L/build.log; echo L8_BUILD_DONE >> $L/build.log
