#!/bin/bash
cd /home/panzihang/src/prism_layerkv
while [ ! -f studies/v29/c_quality/done.json ]; do sleep 120; done
sleep 60
/home/panzihang/venvs/vllm-stable/bin/python scripts/v29_timing_c.py > logs/v29_timing_c.log 2>&1
