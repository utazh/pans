#!/bin/bash
cd /home/panzihang/src/prism_layerkv
PYTHONPATH=src /home/panzihang/venvs/vllm-stable/bin/python scripts/v29_quality_pipeline.py pilot readers --steps 400 --sources syn > logs/v29_quality_pilot_readers_s400.log 2>&1
python3 -c "import json,sys;sys.exit(1 if json.load(open(\"studies/v29/quality/pilot/readers_done_s400.json\"))[\"errors\"] else 0)" || exit 1
PYTHONPATH=src /home/panzihang/venvs/vllm-stable/bin/python scripts/v29_quality_pipeline.py pilot eval --arms syn_s400_G24,syn_s400_G20,syn_s400_G16 > logs/v29_quality_pilot_eval_s400.log 2>&1
