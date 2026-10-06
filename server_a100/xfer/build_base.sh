#!/bin/bash
# Stage 1 on A100: rebuild the ContiguousKV-style base artifacts for one model from scratch, per docs/REPRODUCE_ZH.md.
# Usage: build_base.sh <ModelDir> <tag> <gpu> <probe_query_heads> <selector_kv_heads>
#   e.g. build_base.sh Qwen2.5-7B-Instruct q7 0 0,7,14,21 0,1,2,3
set -euo pipefail
B=/home/shensicheng/qyd/pans; S=$B/src; M=$B/models/$1; T=$2; G=$3; PQH=$4; SKV=$5
ulimit -n 1048576
export TMPDIR=$B/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTHONPATH=$S/src CUDA_VISIBLE_DEVICES=$G
PY=$B/venv/bin/python
D=$B/work/$T; mkdir -p $D
cd $S
echo "== [$T] raw data + datasetwise bundles"
[ -d $D/bundles_full ] || $PY scripts/build_datasetwise_bundles.py --data-root $B/work/data_raw --output-dir $D/bundles_full \
   --tokenizer $M --seed 42 --excluded-uids-manifest configs/calibration_exclusions.json
[ -d $D/bundles_strict ] || $PY scripts/build_strict_eval_bundle.py --source-bundle $D/bundles_full --output-bundle $D/bundles_strict \
   --exclusions configs/strict_eval_exclusions.json
for f in $D/bundles_strict/*.jsonl; do echo "$(wc -l < $f) $(sha256sum $f)"; done
echo "== [$T] prefix store"
[ -d $D/prefix_store ] || $PY -m contiguous_fuxian.sparse_qwen_reprefill prepare --model-path $M --bundle-dir $D/bundles_strict \
   --store-root $D/prefix_store --tasks sst2,subj,trec,rte --samples-per-task 1 --dtype bfloat16 --device cuda
echo "== [$T] pcache c16"
[ -d $D/pcache_c16 ] || $PY -m contiguous_fuxian.flexgen_qwen_reprefill --model-path $M --bundle-dir $D/bundles_strict \
   --store-root $D/prefix_store --plan configs/qwen25_k005_contigkv.json --output-dir $D/build_pcache_run --flexgen-root vendor/flexgen \
   --flexgen-kv-dir $D/pcache_c16 --tasks sst2 --store-tasks sst2,subj,trec,rte --samples-per-task 1 --max-tokens 1 \
   --dtype bfloat16 --device cuda --gpu-cache-mb 0 --cpu-cache-mb 0 --cache-type CKLFU --online-selection \
   --probe-query-heads $PQH --selector-kv-head-ids $SKV --period-size 8 --subperiod-size 4 --expected-keep-ratio 0.05
if [ ! -f $D/pcache_c16/.contiguous_fuxian_complete ]; then   # flexgen 构建不写完成标记；按 v26 p1/v29 schema 补写
  $PY - "$D/pcache_c16" "$SKV" <<'EOF'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1]); heads = [int(h) for h in sys.argv[2].split(',')]
for f in p.iterdir():
    if f.is_file():
        with f.open('rb') as h: os.fsync(h.fileno())
(p / '.contiguous_fuxian_complete').write_text(json.dumps(dict(schema_version=3, method='a100_rebuild_prefix_native_full_kv',
    chunk_size=16, selector_kv_head_ids=heads, online_selection=True, physical_layout='plain-logical-token-order',
    impress_reorder_sha256=None, registered_store_tasks=['sst2', 'subj', 'trec', 'rte'], build_report='build_pcache_run'), indent=2))
print('marker written', heads)
EOF
fi
echo "== [$T] K4 selector index"
[ -d $D/selector_index_k4_g32 ] || $PY scripts/build_selector_index.py --source-pcache-dir $D/pcache_c16 --store-root $D/prefix_store \
   --output-dir $D/selector_index_k4_g32 --model-path $M --tasks sst2,subj,trec,rte --source-task-order sst2,subj,trec,rte \
   --selector-kv-head-ids $SKV --group-size 32
du -sh $D/*
echo BASE_DONE_$T
