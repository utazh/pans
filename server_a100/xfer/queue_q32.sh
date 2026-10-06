#!/bin/bash
# 等 32B 下载完成 → 官方 sha256 校验 → 通过后在 GPU 5 建 32B worktree 与基础产物。
B=/home/shensicheng/qyd/pans
until grep -q MODELS_DONE $B/logs/get_models.log; do sleep 60; done
cd $B/models/Qwen2.5-32B-Instruct && sha256sum -c $B/xfer/sha_32B.txt > $B/logs/sha_32B.log 2>&1
if grep -q -v ": OK$" $B/logs/sha_32B.log; then echo "SHA_32B_FAIL"; cat $B/logs/sha_32B.log; exit 1; fi
echo SHA_32B_OK
# 32B: 40 q heads, 8 kv heads（与 14B 相同的头布局）
bash $B/xfer/make_model.sh q32 Qwen2.5-32B-Instruct 5 0,5,10,15,20,25,30,35 0,1,2,3,4,5,6,7
