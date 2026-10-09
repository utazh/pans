#!/bin/bash
# 改用纯续写格式后重做 Llama：旧（对话模板）产物整体挪到 *_chat_invalid 留档 → setup（转换+基础产物）→ 流水线 → 计时/并发队列。
set -euo pipefail
B=/home/shensicheng/qyd/pans; D=$B/work/l8; X=$B/work/l8_chat_invalid
mkdir -p $X
for d in data v28_frozen bundles_full bundles_strict prefix_store build_pcache_run pcache_c16 selector_index_k4_g32; do
  if [ -e $D/$d ]; then mv $D/$d $X/$d; fi
done
if [ -d $B/src_l8/studies/v29m_s800 ]; then mv $B/src_l8/studies/v29m_s800 $B/src_l8/studies/_v29m_s800_chat_invalid; fi
mv $B/logs/setup_l8.log $X/setup_l8.log 2>/dev/null || true
for f in $B/logs/l8_pipe_*.log; do if [ -e "$f" ]; then mv "$f" $X/; fi; done
setsid nohup bash $B/xfer/setup_l8.sh 0 > $B/logs/setup_l8.log 2>&1 < /dev/null &
sleep 2
setsid nohup bash $B/xfer/queue_l8.sh > $B/logs/queue_l8.log 2>&1 < /dev/null &
sleep 2
setsid nohup bash $B/xfer/queue_after_l8.sh > $B/logs/queue_after_l8.log 2>&1 < /dev/null &
echo REDO_STARTED
