#!/bin/bash
# A100 完整测试集质量评估（v29 链路）。用法: q_eval.sh <gpu> <task> <budget> <name> [adapters_dir]
# 3090 参考（P8）：trec_b25 I=68.60, G16=76.80
B=/home/shensicheng/qyd/pans; S=$B/src; G=$1; T=$2; K=$3; N=$4; AD=${5:-}
export PANS_MODEL_PATH=$B/models/Qwen2.5-7B-Instruct PANS_PY=$B/venv/bin/python CUDA_VISIBLE_DEVICES=$G TMPDIR=$B/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTHONPATH=src
ulimit -n 1048576
cd $S
BUN=studies/v28_frozen/bundles/${T}_b${K}/eval
O=$S/runs_a100/quality/$N; mkdir -p $(dirname $O)
MODE=(--mode identity); [ -n "$AD" ] && MODE=(--mode shared --adapters $AD --allow-candidate --packed-ops)
SAMPLES=$(wc -l < $BUN/$T.jsonl)
$B/venv/bin/python scripts/run_evidence_v26.py --admission reuse-prefetch --audit-jsonl $O.v26.jsonl -- \
  --model-path $B/models/Qwen2.5-7B-Instruct --store-root ${PANS_EVAL_STORE:-runtime_data/prefix_store} --selector-index-dir assets/selector_index_k4_g32 -- \
  --pcache-root ${PANS_EVAL_PCACHE:-runtime_data/pcache_c16} -- "${MODE[@]}" --backend vllm2 --metadata-cache --layout hybrid --stats $O.extra.jsonl -- \
  --task $T --budget $K --period 8 --max-tokens 1 --warmup 0 --samples $SAMPLES --profile none --bundle-dir $BUN --output $O
$B/venv/bin/python - "$O" <<'EOF'
import json, sys, glob
o = sys.argv[1]
s = json.load(open(o + '/summary.json')) if glob.glob(o + '/summary.json') else {}
print({k: v for k, v in s.items() if any(x in k for x in ('accuracy', 'samples', 'ttft', 'bytes'))})
EOF
echo Q_EVAL_DONE $N
