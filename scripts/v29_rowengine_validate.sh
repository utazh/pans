#!/bin/bash
# Row-engine acceptance on trec_b25: (1) warm parity, 32 test queries, vs pcache path / quality bridge;
# (2) one cold block per arm on GPU2 (cores 28-31,60-63), cold certification on the files actually read.
R=/home/panzihang/src/prism_layerkv; cd $R; P=/home/panzihang/venvs/vllm-stable/bin/python
O=$R/studies/v29/rowengine_validate; mkdir -p $O
export PYTHONPATH=src:runtime_deps_v24 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
G16=$R/studies/v28_frozen/readers/trec_b25
run () { # gpu cores name engine datadir mode adapters samples warmup bundle cold
  local o=$O/$3; local ad=""; [ "$7" != "-" ] && ad="--adapters $7 --allow-candidate --packed-ops"; local c=""; [ "${11}" = cold ] && c="--row-cold"
  CUDA_VISIBLE_DEVICES=$1 taskset -c $2 $P scripts/v29_rowengine_entry.py --engine $4 --data-dir $5 $c --row-log $o.row.json \
    --admission reuse-prefetch --audit-jsonl $o.v26.jsonl -- --model-path /data1/llm/Qwen/Qwen2.5-7B-Instruct \
    --store-root $R/runtime_data/prefix_store --selector-index-dir $R/assets/selector_index_k4_g32 -- --pcache-root $R/runtime_data/v29_pcache_gpu2_ssd -- \
    --mode $6 --backend vllm2 --metadata-cache --layout hybrid --stats $o.extra.jsonl $ad -- --task trec --budget 25 --period 8 \
    --max-tokens 1 --warmup $9 --samples $8 --profile none --bundle-dir ${10} --output $o > $o.log 2>&1
  echo "$3 rc=$?"
}
E=$R/studies/v28_frozen/bundles/trec_b25/eval; B0=$R/studies/v28_frozen/bundles/trec_b25/b00
run 0 0-7,32-39 par_I native $R/runtime_data/v29_rows_ssd/trec identity - 32 0 $E warm &
run 1 8-15,40-47 par_S native $R/runtime_data/v29_rows_ssd/trec shared $G16 32 0 $E warm &
run 3 48-59 par_X xkv $R/runtime_data/v29_xkv_ssd/trec_b25 identity - 32 0 $E warm &
run 2 28-31,60-63 cold_I native $R/runtime_data/v29_rows_ssd/trec identity - 16 4 $B0 cold
run 2 28-31,60-63 cold_S native $R/runtime_data/v29_rows_ssd/trec shared $G16 16 4 $B0 cold
run 2 28-31,60-63 cold_X xkv $R/runtime_data/v29_xkv_ssd/trec_b25 identity - 16 4 $B0 cold
wait
$P - <<'EOF'
import json, statistics as st
R = '/home/panzihang/src/prism_layerkv/studies'
def recs(p): return {r['uid']: r for r in map(json.loads, open(p + '/scored_records.jsonl'))}
for name, ref in (('par_I', 'v29/test_quality/runs/test_I_trec_b25'), ('par_S', 'v29/test_quality/runs/test_G16_trec_b25'),
                  ('par_X', 'v28_frozen/runs/trec_b25_external_quality_clf_256_384_bf16')):
    A, B = recs(f'{R}/v29/rowengine_validate/{name}'), recs(f'{R}/{ref}')
    d = max(max(abs(A[u]['label_candidate_mean_logprobs'][k] - B[u]['label_candidate_mean_logprobs'][k]) for k in A[u]['label_candidate_mean_logprobs']) for u in A)
    print(f'{name} vs {ref.split("/")[-1]}: n={len(A)} max|dscore|={d} same_pred={sum(A[u]["prediction"]==B[u]["prediction"] for u in A)} '
          f'same_selection={sum(A[u]["layer_token_selection_sha256"]==B[u]["layer_token_selection_sha256"] for u in A)}')
for name in ('cold_I', 'cold_S', 'cold_X'):
    r = [json.loads(l) for l in open(f'{R}/v29/rowengine_validate/{name}/scored_records.jsonl')]
    x = json.load(open(f'{R}/v29/rowengine_validate/{name}.row.json')); m = [q for q in x['requests'] if q['measured']]
    b = x.get('request_bytes', [])
    print(f'{name}: TTFT {st.mean(a["ttft_ms"] for a in r):.1f} ms | cache_update_ms {st.mean(a.get("cache_update_ms",0) for a in r):.1f} | '
          f'cold certs {len(m)} resident={sorted(set(q["cold_resident_after"] for q in m))} files={m[0]["cold_files"]} | MB/request {st.mean(b[-16:])/1e6:.1f}')
EOF
