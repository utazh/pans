#!/bin/bash
# Llama-3.1-8B-Instruct 在 A100 上的完整准备：模型 -> worktree+补丁 -> Llama 模板数据 -> 基础产物。可重复执行（已完成的步骤跳过）。
# 用法: setup_l8.sh <gpu>
set -euo pipefail
B=/home/shensicheng/qyd/pans; T=l8; M=Llama-3.1-8B-Instruct; G=$1
PQH=0,4,8,12,16,20,24,28; SKV=0,1,2,3,4,5,6,7          # 32 q 头 / 8 KV 头：每个 KV 组取第一个 q 头作探针
export TMPDIR=$B/tmp; PY=$B/venv/bin/python; MD=$B/models/$M; D=$B/work/$T; WT=$B/src_$T
mkdir -p $TMPDIR $D

echo "== [$T] model (wuping 只读复制 + ModelScope 补第 1 分片；全部按 ModelScope sha256 校验)"
if [ ! -f $MD/.verified ]; then
  mkdir -p $MD
  for f in config.json generation_config.json model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer_config.json \
           model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors LICENSE; do
    [ -f $MD/$f ] || cp /home/wuping/model/$M/$f $MD/$f
  done
  [ -f $MD/model-00001-of-00004.safetensors ] || curl -sL --retry 5 -C - -o $MD/model-00001-of-00004.safetensors \
     https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct/resolve/master/model-00001-of-00004.safetensors
  cat > $TMPDIR/l8.sha <<'EOF'
29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e  config.json
189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd  generation_config.json
2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668  model-00001-of-00004.safetensors
09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15  model-00002-of-00004.safetensors
fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa  model-00003-of-00004.safetensors
92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b  model-00004-of-00004.safetensors
79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4  tokenizer.json
177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424  tokenizer_config.json
EOF
  (cd $MD && sha256sum -c $TMPDIR/l8.sha) && touch $MD/.verified
fi

echo "== [$T] worktree + Llama 补丁"
if [ ! -d $WT ]; then
  cd $B/src && git worktree add -q -b a100-$T $WT a100
  $PY $B/xfer/llama_patch.py $WT
fi

echo "== [$T] Llama 模板数据副本（原数据不动）"
[ -d $D/data ] || $PY $B/xfer/convert_llama.py $B/src/data $D/data
[ -d $D/v28_frozen/bundles ] || $PY $B/xfer/convert_llama.py $B/src/studies/v28_frozen/bundles $D/v28_frozen/bundles
[ -e $D/v28_frozen/readers ] || ln -s $B/src/studies/v28_frozen/readers $D/v28_frozen/readers
[ -d $D/bundles_full ] || $PY $B/xfer/convert_llama.py $B/work/q7/bundles_full $D/bundles_full
[ -d $D/bundles_strict ] || $PY $B/xfer/convert_llama.py $B/work/q7/bundles_strict $D/bundles_strict

cd $WT
ln -sfn $D/data data; mkdir -p studies; ln -sfn $D/v28_frozen studies/v28_frozen; ln -sfn $B/src/audit audit

echo "== [$T] 基础产物（prefix store / pcache / K4 index）"
bash $B/xfer/build_base.sh $M $T $G $PQH $SKV
mkdir -p runtime_data assets
ln -sfn $D/prefix_store runtime_data/prefix_store
ln -sfn $D/pcache_c16 runtime_data/pcache_c16
ln -sfn $D/selector_index_k4_g32 assets/selector_index_k4_g32
echo "export PANS_MODEL_PATH=$MD PANS_PY=$PY PANS_PROBE_QUERY_HEADS=$PQH PANS_SELECTOR_KV_HEAD_IDS=$SKV TMPDIR=$B/tmp" > $B/env_$T.sh
echo MODEL_READY_$T
