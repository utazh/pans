#!/bin/bash
# P1-a/b numeric validation (32 test queries, trec_b25 G16 and sst2_b25 G20, GPU0):
#  direct-exec (no reader) vs unit-reader checkpoint;  explicit-reconstruction vs read-side (same checkpoint).
set -e
R=/home/panzihang/src/prism_layerkv; cd $R; PY=/home/panzihang/venvs/vllm-stable/bin/python
O=$R/studies/v29/p1ab_validate; mkdir -p $O
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:runtime_deps_v24 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
# unit-reader checkpoints from the same R0/selected checkpoints (existing sharedkv_v26.direct.export_direct)
for spec in trec_b25:G16 sst2_b25:G20; do
  cell=${spec%%:*}; g=${spec##*:}; task=${cell%%_*}
  src=$($PY -c "import json;print(json.load(open('studies/v29/readers/manifest.json'))['cells']['$cell']['$g']['adapter_dir'])")
  dst=studies/v29/readers/${cell}_${g}_unit
  [ -e $dst/$task.pt ] || $PY -c "from contiguous_fuxian.sharedkv_v26.direct import export_direct;export_direct('$src/$task.pt','$dst/$task.pt')"
  echo "$cell $g src=$src unit=$dst"
  run () { # name launcher adapters
    o=$O/$1; [ -e $o ] && return
    taskset -c 0-7,32-39 $PY $2 --admission reuse-prefetch --audit-jsonl $o.v26.jsonl -- --model-path /data1/llm/Qwen/Qwen2.5-7B-Instruct \
      --store-root $R/runtime_data/prefix_store --selector-index-dir $R/assets/selector_index_k4_g32 -- --pcache-root $R/runtime_data/v27_pcache_gpu3 -- \
      --mode shared --backend vllm2 --metadata-cache --layout hybrid --stats $o.extra.jsonl --adapters $3 --allow-candidate --packed-ops -- \
      --task $task --budget 25 --period 8 --max-tokens 1 --warmup 0 --samples 32 --profile none \
      --bundle-dir $R/studies/v28_frozen/bundles/$cell/eval --output $o > $o.log 2>&1
    echo "  $1 rc=$? $(grep -o 'V29_EXEC_VARIANT.*' $o.log)"
  }
  run ${cell}_readside "scripts/run_evidence_v26.py" $src
  run ${cell}_unitreader "scripts/run_evidence_v26.py" $R/$dst
  run ${cell}_direct "scripts/v29_exec_variant.py --variant direct" $src
  run ${cell}_explicit "scripts/v29_exec_variant.py --variant explicit" $src
done
$PY - <<'EOF'
import json, math
O = '/home/panzihang/src/prism_layerkv/studies/v29/p1ab_validate'
def load(n): return {r['uid']: r for r in map(json.loads, open(f'{O}/{n}/scored_records.jsonl'))}
for cell in ['trec_b25', 'sst2_b25']:
    for a, b in [('direct', 'unitreader'), ('explicit', 'readside')]:
        A, B = load(f'{cell}_{a}'), load(f'{cell}_{b}')
        d = [max(abs(A[u]['label_candidate_mean_logprobs'][k] - B[u]['label_candidate_mean_logprobs'][k]) for k in A[u]['label_candidate_mean_logprobs']) for u in A]
        same_pred = sum(A[u]['prediction'] == B[u]['prediction'] for u in A)
        sel = sum(A[u]['layer_token_selection_sha256'] == B[u]['layer_token_selection_sha256'] for u in A)
        print(f'{cell} {a:8s} vs {b:10s}: n={len(A)} max|dscore|={max(d):.4f} mean={sum(d)/len(d):.4f} same_pred={same_pred} same_selection={sel} '
              f'acc {sum(r["correct"] for r in A.values())} vs {sum(r["correct"] for r in B.values())}')
EOF
