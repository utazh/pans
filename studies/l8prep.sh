#!/bin/bash
# Llama-3.1-8B on the 3090: verify weights against the official sha256 list, make an independent working copy of the
# A100 Llama worktree (backup) on /data1, and repoint its symlinks (KV stores are rebuilt later).
set -u
B=/data1/pans_l8; mkdir -p $B/logs; L=$B/logs/prep.log; M=/data1/llm/meta/Meta-Llama-3.1-8B-Instruct
mkdir $B/lock_prep 2>/dev/null || exit 0
echo "== sha $(date)" >> $L
( cd $M && grep -E "safetensors|config.json|tokenizer" /home/panzihang/src/pans/tmp/l8.sha | sha256sum -c - ) >> $L 2>&1 && echo SHA_OK >> $L || echo SHA_FAIL >> $L
echo "== copy $(date)" >> $L
rsync -a /home/panzihang/src/pans/src_l8/ $B/raps_l8/ && rsync -a /home/panzihang/src/pans/work/l8/ $B/work_l8/ && echo COPY_OK >> $L
cd $B/raps_l8
for l in assets/selector_index_k4_g32 data studies/v28_frozen runtime_data/prefix_store runtime_data/pcache_c16 runtime_data/v29m_pcache_timing; do
  echo "$l -> $(readlink $l)" >> $L
done
ln -sfn $B/work_l8/selector_index_k4_g32 assets/selector_index_k4_g32
ln -sfn $B/work_l8/data data
ln -sfn $B/work_l8/v28_frozen studies/v28_frozen
ln -sfn $B/work_l8/prefix_store runtime_data/prefix_store
ln -sfn $B/work_l8/pcache_c16 runtime_data/pcache_c16
rm -f runtime_data/v29m_pcache_timing
echo LINKS_OK >> $L; ls -la assets/selector_index_k4_g32 data studies/v28_frozen runtime_data/prefix_store runtime_data/pcache_c16 >> $L
echo PREP_DONE >> $L
