#!/bin/bash
# QuALITY formal 12 articles (f0,f1,f2), 25%, frozen 200-step synthetic-calibration recipe; native sharing only.
cd /home/panzihang/src/prism_layerkv
P=/home/panzihang/venvs/vllm-stable/bin/python
set -e
i=0; for s in f0 f1 f2; do CUDA_VISIBLE_DEVICES=$i $P scripts/prepare_v29_quality.py synth $s > logs/v29_quality_synth_$s.log 2>&1 & i=$((i+1)); done; wait
i=0; for s in f0 f1 f2; do CUDA_VISIBLE_DEVICES=$i PYTHONPATH=src $P scripts/v29_quality_pipeline.py $s build > logs/v29_quality_${s}_build.log 2>&1 & i=$((i+1)); done; wait
for s in f0 f1 f2; do
  PYTHONPATH=src $P scripts/v29_quality_pipeline.py $s readers --sources syn > logs/v29_quality_${s}_readers.log 2>&1
  python3 -c "import json,sys;sys.exit(1 if json.load(open(\"studies/v29/quality/$s/readers_done.json\"))[\"errors\"] else 0)"
done
for s in f0 f1 f2; do PYTHONPATH=src $P scripts/v29_quality_pipeline.py $s eval --arms I,FULLKV,syn_G24,syn_G20,syn_G16 > logs/v29_quality_${s}_eval.log 2>&1; done
i=0; for s in f0 f1 f2; do CUDA_VISIBLE_DEVICES=$i PYTHONPATH=src $P scripts/v29_quality_pipeline.py $s nodoc > logs/v29_quality_${s}_nodoc.log 2>&1 & i=$((i+1)); done; wait
echo FORMAL_DONE > studies/v29/quality/formal_done.txt
