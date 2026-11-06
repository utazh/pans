#!/bin/bash
cd /home/panzihang/src/prism_layerkv
P=/home/panzihang/venvs/vllm-stable/bin/python
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $P scripts/v29_subj_p4.py readers > logs/v29_subj_p4_readers.log 2>&1 || exit 1
PYTHONPATH=src $P scripts/v29_subj_p4.py eval > logs/v29_subj_p4_eval.log 2>&1
