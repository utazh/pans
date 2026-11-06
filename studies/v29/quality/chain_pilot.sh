#!/bin/bash
cd /home/panzihang/src/prism_layerkv
while [ ! -f studies/v29/quality/pilot/readers_done.json ]; do sleep 60; done
python3 -c "import json,sys;sys.exit(1 if json.load(open(\"studies/v29/quality/pilot/readers_done.json\"))[\"errors\"] else 0)" || exit 1
PYTHONPATH=src /home/panzihang/venvs/vllm-stable/bin/python scripts/v29_quality_pipeline.py pilot eval > logs/v29_quality_pilot_eval.log 2>&1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/panzihang/venvs/vllm-stable/bin/python scripts/v29_quality_pipeline.py pilot nodoc > logs/v29_quality_pilot_nodoc.log 2>&1
