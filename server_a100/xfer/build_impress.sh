#!/bin/bash
# 为一个模型构建 IMPRESS 资产（与 3090 v29 相同流程）。用法: build_impress.sh <tag q7|q14|q32> <gpu>
#  1) 重排 manifest：impress_reorder，bundle = data/v26/p0/fit，每任务 4 条（offset 0），BF16
#  2) c64 Pcache：run_round4_methods.py --method impress --build-kv（sst2_b5 两条请求，注册全部 4 个任务）
#  3) 选择器 key 文件：校验已有 + 补齐缺失；4) 完成标记（schema 2）；5) 计时用独立副本
set -euo pipefail
B=/home/shensicheng/qyd/pans; T=$1; G=$2
if [ $T = q7 ]; then ROOT=$B/src; export PANS_MODEL_PATH=$B/models/Qwen2.5-7B-Instruct PANS_SELECTOR_KV_HEAD_IDS=0,1,2,3; else ROOT=$B/src_$T; . $B/env_$T.sh; fi
export CUDA_VISIBLE_DEVICES=$G TMPDIR=$B/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTHONPATH=src
ulimit -n 1048576
PY=$B/venv/bin/python; D=$B/work/$T; MAN=$D/impress_reorder_fit4.json; C64=$D/pcache_impress_c64
cd $ROOT
[ -f $MAN ] || $PY -m contiguous_fuxian.impress_reorder --model-path $PANS_MODEL_PATH --bundle-dir data/v26/p0/fit \
   --store-root runtime_data/prefix_store --samples-per-task 4 --sample-offset 0 --device cuda --allow-gpu --dtype bfloat16 --output $MAN
SHA=$(sha256sum $MAN | cut -d" " -f1); echo "manifest sha256 $SHA"
if [ ! -f $C64/.contiguous_fuxian_complete ]; then
  rm -rf $ROOT/studies/v29m_impress_build_$T
  $PY scripts/run_round4_methods.py --method impress --build-kv --kv-dir $C64 --impress-reorder $MAN --task sst2 --budget 5 \
     --samples 2 --warmup 0 --bundle-dir studies/v28_frozen/bundles/sst2_b5/eval --output $ROOT/studies/v29m_impress_build_$T
  $PY $B/xfer/fill_impress_head.py $ROOT $C64 $MAN --write
  for f in $C64/*; do :; done
  $PY - "$C64" "$SHA" <<'EOF'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1])
for f in p.iterdir():
    if f.is_file():
        with f.open('rb') as h: os.fsync(h.fileno())
(p / '.contiguous_fuxian_complete').write_text(json.dumps(dict(schema_version=2, method='impress', chunk_size=64, selector_kv_head_ids=[0],
    online_selection=True, impress_reorder_sha256=sys.argv[2], built_by='scripts/run_round4_methods.py --build-kv (A100 v29m)',
    prefix_store='runtime_data/prefix_store'), indent=2))
print('marker written')
EOF
fi
[ -d $D/pcache_impress_c64_timing ] || cp -a $C64 $D/pcache_impress_c64_timing
du -sh $C64 $D/pcache_impress_c64_timing
echo IMPRESS_READY_$T
