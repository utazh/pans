#!/bin/bash
cd /home/panzihang/src/prism_layerkv
/home/panzihang/venvs/vllm-stable/bin/python scripts/v29_p1ab.py quality > logs/v29_p1ab_quality.log 2>&1
sleep 60
/home/panzihang/venvs/vllm-stable/bin/python scripts/v29_p1ab.py timing > logs/v29_timing_p1ab.log 2>&1
