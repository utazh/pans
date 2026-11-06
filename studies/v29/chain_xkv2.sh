#!/bin/bash
cd /home/panzihang/src/prism_layerkv
P=/home/panzihang/venvs/vllm-stable/bin/python
$P scripts/v29_xkv_dev.py run > logs/v29_xkv_dev.log 2>&1
grep -v "returncode\": 0" studies/v29/xkv_dev/status.jsonl && exit 1
$P scripts/v29_xkv_dev.py select > logs/v29_xkv_dev_select.log 2>&1
sleep 60
$P scripts/v29_timing_row2.py > logs/v29_timing_row2.log 2>&1
