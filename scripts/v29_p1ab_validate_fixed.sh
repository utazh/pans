#!/bin/bash
# P1-a/b arithmetic validation with a FIXED selection plan (isolates the mechanism from online-selection drift).
# 1) record selection + first-token logits from the read-side run; 2) rerun read-side / explicit / unit-reader /
# direct with --selection-plan (diagnostic) and --logits; compare full-vocab first-token logits and label scores.
set -e
R=/home/shensicheng/qyd/pans/src; cd $R; PY=/home/shensicheng/qyd/pans/venv/bin/python
O=$R/studies/v29/p1ab_validate_fixed; mkdir -p $O
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:runtime_deps_v24 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
for spec in trec_b25:G16 sst2_b25:G20; do
  cell=${spec%%:*}; g=${spec##*:}; task=${cell%%_*}
  src=$($PY -c "import json;print(json.load(open('studies/v29/readers/manifest.json'))['cells']['$cell']['$g']['adapter_dir'])")
  unit=$R/studies/v29/readers/${cell}_${g}_unit
  run () { # name launcher adapters extra_v21_args
    o=$O/$1; [ -e $o ] && return
    taskset -c 0-7,32-39 $PY $2 --admission reuse-prefetch --audit-jsonl $o.v26.jsonl -- --model-path /home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct \
      --store-root $R/runtime_data/prefix_store --selector-index-dir $R/assets/selector_index_k4_g32 -- --pcache-root $R/runtime_data/v27_pcache_gpu3 -- \
      --mode shared --backend vllm2 --metadata-cache --layout hybrid --stats $o.extra.jsonl --adapters $3 --allow-candidate --packed-ops \
      --diagnostic --logits $o.logits $4 -- \
      --task $task --budget 25 --period 8 --max-tokens 1 --warmup 0 --samples 32 --profile none \
      --bundle-dir $R/studies/v28_frozen/bundles/$cell/eval --output $o > $o.log 2>&1
    echo "  $1 rc=$?"
  }
  run ${cell}_plan "scripts/run_evidence_v26.py" $src ""
  P="--selection-plan $O/${cell}_plan.logits"
  run ${cell}_readside "scripts/run_evidence_v26.py" $src "$P"
  run ${cell}_explicit "scripts/v29_exec_variant.py --variant explicit" $src "$P"
  run ${cell}_unitreader "scripts/run_evidence_v26.py" $unit "$P"
  run ${cell}_direct "scripts/v29_exec_variant.py --variant direct" $src "$P"
done
$PY - <<'EOF'
import json, glob, torch
O = '/home/shensicheng/qyd/pans/src/studies/v29/p1ab_validate_fixed'
def recs(n): return {r['uid']: r for r in map(json.loads, open(f'{O}/{n}/scored_records.jsonl'))}
def logits(n): return {torch.load(p, map_location='cpu', weights_only=False)['query_hash']: torch.load(p, map_location='cpu', weights_only=False) for p in glob.glob(f'{O}/{n}.logits/*.pt')}
for cell in ['trec_b25', 'sst2_b25']:
    for a, b in [('readside', 'plan'), ('explicit', 'readside'), ('direct', 'unitreader')]:
        A, B = recs(f'{cell}_{a}'), recs(f'{cell}_{b}'); LA, LB = logits(f'{cell}_{a}'), logits(f'{cell}_{b}')
        sel = sum(A[u]['layer_token_selection_sha256'] == B[u]['layer_token_selection_sha256'] for u in A)
        d = [max(abs(A[u]['label_candidate_mean_logprobs'][k] - B[u]['label_candidate_mean_logprobs'][k]) for k in A[u]['label_candidate_mean_logprobs']) for u in A]
        dl = [float((LA[h]['logits'].float() - LB[h]['logits'].float()).abs().max()) for h in LA]
        rel = [float((LA[h]['logits'].float() - LB[h]['logits'].float()).norm() / LB[h]['logits'].float().norm()) for h in LA]
        top = sum(int(LA[h]['logits'].argmax()) == int(LB[h]['logits'].argmax()) for h in LA)
        print(f'{cell} {a:9s} vs {b:10s}: same_selection {sel}/{len(A)} | first-token logits max|d| {max(dl):.4f} rel-L2 max {max(rel):.2e} top1 {top}/{len(LA)} '
              f'| label score max|d| {max(d):.4f} mean {sum(d)/len(d):.4f} | same_pred {sum(A[u]["prediction"]==B[u]["prediction"] for u in A)}/{len(A)}')
EOF
