#!/bin/bash
# Llama 计时结束后：观察 3（相邻层 KV 角距离；读取器前后误差），SST-2 与 TREC，S3 读取器。
B=/home/shensicheng/qyd/pans
until grep -q L8_TIMING_DONE $B/logs/a100_sys_l8.log; do sleep 60; done
cd $B; export TMPDIR=$B/tmp CUDA_VISIBLE_DEVICES=4
for t in sst2 trec; do
  venv/bin/python xfer/kv_similarity.py work/l8/prefix_store $t analysis/kvsim_l8_$t.json src_l8/studies/v29m_s800/readers/${t}_b25/S3/$t.pt
done
echo KVSIM_L8_DONE
