#!/bin/bash
# 14B/32B 基线完整测试集质量（25%）：ContiguousKV / IMPRESS，与 3090 v29_c_quality.py 相同参数（v28 冻结测试 bundle，warmup 0）。
# 用法: sysquality.sh <tag q14|q32> <gpu> <cores> <method contigkv|impress> <task...>
B=/home/shensicheng/qyd/pans; T=$1; G=$2; C=$3; M=$4; shift 4
if [ $T = q7 ]; then ROOT=$B/src; export PANS_MODEL_PATH=$B/models/Qwen2.5-7B-Instruct PANS_SELECTOR_KV_HEAD_IDS=0,1,2,3 PANS_PY=$B/venv/bin/python; else ROOT=$B/src_$T; . $B/env_$T.sh; fi
export CUDA_VISIBLE_DEVICES=$G TMPDIR=$B/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTHONPATH=src:runtime_deps_v24
ulimit -n 1048576
cd $ROOT; OUT=$ROOT/studies/v29m_sys/quality; mkdir -p $OUT/runs $OUT/logs
for task in "$@"; do
  name=${M}_${task}_b25; o=$OUT/runs/$name
  [ -f $o/summary.json ] && { echo "skip $name"; continue; }
  ts=$(date +%s); for x in $o $o.*; do [ -e "$x" ] && mkdir -p $OUT/runs/_aborted_${name}_$ts && mv "$x" $OUT/runs/_aborted_${name}_$ts/; done
  n=$(wc -l < studies/v28_frozen/bundles/${task}_b25/eval/$task.jsonl)
  if [ $M = impress ]; then kv=$B/work/$T/pcache_impress_c64; extra="--impress-reorder $B/work/$T/impress_reorder_fit4.json"; else kv=$ROOT/runtime_data/pcache_c16; extra=""; fi
  t0=$(date +%s)
  taskset -c $C $B/venv/bin/python scripts/v29_baseline_entry.py --admission original --audit-jsonl $o.v26.jsonl -- \
    --model-path $PANS_MODEL_PATH --store-root runtime_data/prefix_store -- --pcache-root $kv -- --mode original -- \
    --method $M $extra --task $task --budget 25 --warmup 0 --samples $n --bundle-dir studies/v28_frozen/bundles/${task}_b25/eval \
    --output $o > $OUT/logs/$name.$(date +%s).log 2>&1
  echo "{\"name\": \"$name\", \"rc\": $?, \"seconds\": $(( $(date +%s) - t0 ))}" | tee -a $OUT/status.jsonl
done
echo SYSQ_DONE_${T}_${M}
