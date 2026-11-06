#!/bin/bash
cd /home/panzihang/src/prism_layerkv
while [ ! -f studies/v29/p0_test_g/done.json ]; do sleep 60; done
python3 scripts/v29_crosslayer_report.py > logs/v29_crosslayer_report.log 2>&1
sleep 60
/home/panzihang/venvs/vllm-stable/bin/python scripts/v29_timing_p1c.py > logs/v29_timing_p1c.log 2>&1
