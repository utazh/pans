#!/bin/bash
# after v30_qa_p4.py finishes: QuALITY cold timing
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v30/logs
until [ -f studies/v30/qa_p4/done.json ]; do sleep 60; done
$P scripts/v30_timing_qa.py > $L/timing_qa.log 2>&1
echo CHAIN_QA_DONE >> $L/timing_qa.log
