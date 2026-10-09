#!/bin/bash
# Llama 计时前准备：计时用 pcache 独立副本 + IMPRESS 资产（重排 manifest、c64 pcache、计时副本）。用法: prep_l8_timing.sh <gpu>
set -euo pipefail
B=/home/shensicheng/qyd/pans; D=$B/work/l8
[ -f $D/pcache_timing/.contiguous_fuxian_complete ] || cp -a $D/pcache_c16 $D/pcache_timing
ln -sfn $D/pcache_timing $B/src_l8/runtime_data/v29m_pcache_timing
cp $B/xfer/v29m_timing_sys.py $B/src_l8/scripts/v29m_timing_sys.py
bash $B/xfer/build_impress.sh l8 $1
echo L8_TIMING_PREP_DONE
