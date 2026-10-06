#!/bin/bash
# 等所有 v29m 质量流程（会持续读盘）结束后，再启动 7B 正式计时（GPU 6），避免磁盘干扰。
B=/home/shensicheng/qyd/pans
while pgrep -u shensicheng -f "scripts/v29m_pipeline.py" > /dev/null; do sleep 120; done
sleep 60
cd $B/src && . $B/env_timing.sh
exec $B/venv/bin/python scripts/v29a_timing.py a100_timing_q7
