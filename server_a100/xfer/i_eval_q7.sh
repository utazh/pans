#!/bin/bash
B=/home/shensicheng/qyd/pans
for t in sst2 rte subj; do bash $B/xfer/q_eval.sh 4 $t 25 q7_${t}_b25_I > $B/logs/q_eval_q7_${t}_b25_I.log 2>&1; done
for t in trec sst2 rte subj; do O=$B/src/runs_a100/quality/q7_${t}_b25_I; $B/venv/bin/python -c "
import json; R=[json.loads(l) for l in open(\"$O/scored_records.jsonl\")]; print(\"$t\", len(R), round(100*sum(r[\"correct\"] for r in R)/len(R),2))"; done
echo I_EVAL_DONE
