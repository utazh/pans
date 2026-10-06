#!/bin/bash
B=/home/shensicheng/qyd/pans; cd $B/src; ulimit -n 1048576
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src TMPDIR=$B/tmp taskset -c 64-99,192-227 $B/venv/bin/python $B/xfer/minicache_build.py runtime_data/prefix_store $B/work/q7/mcctrl_store $B/work/q7/mcctrl_pcache --retain 1.0 || exit 1
PANS_EVAL_STORE=$B/work/q7/mcctrl_store PANS_EVAL_PCACHE=$B/work/q7/mcctrl_pcache bash $B/xfer/q_eval.sh 6 trec 25 q7_trec_b25_mc_retainall > $B/logs/q_eval_mcctrl.log 2>&1
O=$B/src/runs_a100/quality/q7_trec_b25_mc_retainall; I=$B/src/runs_a100/quality/q7_trec_b25_I
$B/venv/bin/python -c "
import json
R={json.loads(l)[\"uid\"]:json.loads(l) for l in open(\"$O/scored_records.jsonl\")}; I={json.loads(l)[\"uid\"]:json.loads(l) for l in open(\"$I/scored_records.jsonl\")}
print(\"ctrl acc\", 100*sum(r[\"correct\"] for r in R.values())/len(R), \"pred-identical to I:\", sum(R[u][\"prediction\"]==I[u][\"prediction\"] for u in R), \"/\", len(R))"
echo MC_CTRL_DONE
