#!/bin/bash
# v35 QuALITY baselines (7B, 12 articles, 25 %): ContiguousKV / IMPRESS / AS+H2O+LRU quality on all real questions.
# Runs on GPUs 0,1 from HDD copies (/data1/pans_qa) while GPU 2 does SSD timing. Per article set a mini root (symlinks to the
# 7B tree; runtime_data/prefix_store -> that set's store) lets the unchanged builders run: IMPRESS reorder manifest
# (A100 tree module, 4 synthetic calibration questions per slot = the "fit4" rule), c64 reordered Pcache + selector head
# files + marker (as xfer/build_impress.sh), AS c64 plain Pcache; then quality through the v26 chain with store override.
set -u
Q=/data1/pans_qa; T=/home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; M=/data1/llm/Qwen/Qwen2.5-7B-Instruct
mkdir -p $Q/logs; mkdir $Q/lock 2>/dev/null || exit 0
until [ -f $Q/copy.done ]; do sleep 30; done
ulimit -n 1048576
one_set() {   # $1 = set, $2 = gpu
  s=$1; g=$2; root=$Q/root_$s; L=$Q/logs/$s.log; export CUDA_VISIBLE_DEVICES=$g PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 TMPDIR=$Q/tmp; mkdir -p $TMPDIR
  mkdir -p $root/scripts $root/runtime_data $root/studies
  for x in src vendor configs data runtime_deps_v24; do [ -e $root/$x ] || ln -s $T/$x $root/$x; done
  ln -sfn $Q/v29_quality_${s}_store $root/runtime_data/prefix_store
  cp $T/scripts/run_round4_methods_as.py $root/scripts/
  MAN=$Q/impress_reorder_$s.json; C64=$Q/imp_c64_$s; CAS=$Q/as_c64_$s
  [ -f $MAN ] || PYTHONPATH=/home/panzihang/src/pans/src/src $P -m contiguous_fuxian.impress_reorder --model-path $M --bundle-dir $T/data/v29/quality/$s/syn_fit \
      --store-root $Q/v29_quality_${s}_store --samples-per-task 4 --sample-offset 0 --device cuda --allow-gpu --dtype bfloat16 --output $MAN >> $L 2>&1
  if [ ! -f $C64/.contiguous_fuxian_complete ]; then
    ( cd $root && PYTHONPATH=src:runtime_deps_v24 PANS_MODEL_PATH=$M $P scripts/run_round4_methods_as.py --method impress --build-kv --kv-dir $C64 --impress-reorder $MAN \
        --task sst2 --budget 5 --samples 2 --warmup 0 --bundle-dir $T/data/v29/quality/$s/eval --output $root/studies/imp_build ) >> $L 2>&1
    $P /home/panzihang/src/pans/xfer/fill_impress_head.py $root $C64 $MAN --write >> $L 2>&1
    $P - "$C64" "$(sha256sum $MAN | cut -d' ' -f1)" >> $L 2>&1 <<'PYEOF'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1])
for f in p.iterdir():
    if f.is_file():
        with f.open('rb') as h: os.fsync(h.fileno())
(p / '.contiguous_fuxian_complete').write_text(json.dumps(dict(schema_version=2, method='impress', chunk_size=64, selector_kv_head_ids=[0],
    online_selection=True, impress_reorder_sha256=sys.argv[2], built_by='run_round4_methods --build-kv (3090 v35 QuALITY)', prefix_store='runtime_data/prefix_store'), indent=2))
print('marker written')
PYEOF
  fi
  [ -d $CAS ] || ( cd $root && PYTHONPATH=src:runtime_deps_v24 PANS_MODEL_PATH=$M $P scripts/run_round4_methods_as.py --method as_h2o_lru --build-kv --kv-dir $CAS \
        --task sst2 --budget 25 --samples 2 --warmup 0 --bundle-dir $T/data/v29/quality/$s/eval --output $root/studies/as_build ) >> $L 2>&1
  cd $T
  for slot in sst2 subj trec rte; do
    n=$(grep -c . $T/data/v29/quality/$s/eval/$slot.jsonl)
    for m in CKV IMP AS; do
      o=$T/studies/v35/qa_baselines/runs/${s}_${slot}_$m; [ -f $o/summary.json ] && continue; mkdir -p $(dirname $o)
      case $m in CKV) kv=$Q/v29_quality_${s}_pcache; meth="--method contigkv"; e=v29_baseline_entry.py;;
                 IMP) kv=$C64; meth="--method impress --impress-reorder $MAN"; e=v29_baseline_entry.py;;
                 AS)  kv=$CAS; meth="--method as_h2o_lru"; e=v33_as_entry.py;; esac
      PYTHONPATH=src:runtime_deps_v24 taskset -c $( [ $g = 0 ] && echo 32-39 || echo 40-47 ) $P scripts/$e --admission original --audit-jsonl $o.v26.jsonl -- \
        --model-path $M --store-root $Q/v29_quality_${s}_store -- --pcache-root $kv -- --mode original -- $meth --task $slot --budget 25 --max-tokens 1 \
        --warmup 0 --samples $n --bundle-dir $T/data/v29/quality/$s/eval --output $o >> $L 2>&1
      echo "{\"name\": \"${s}_${slot}_$m\", \"rc\": $?, \"t\": $(date +%s)}" >> $Q/status.jsonl
    done
  done
}
( one_set f0 0; one_set f2 0 ) &
( one_set f1 1 ) &
wait; echo QA_BASELINES_DONE >> $Q/status.jsonl
