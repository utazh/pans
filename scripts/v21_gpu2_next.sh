#!/bin/bash
set -euo pipefail
cd /home/panzihang/src/prism_layerkv
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=/home/panzihang/venvs/vllm-stable/bin/python
NSYS=/home/panzihang/src/prism_layerkv/runtime_tools/nsight-2025.5.1/opt/nvidia/nsight-systems-cli/2025.5.1/target-linux-x64/nsys
while [ ! -f results/v21_reg_dense7/summary.json ]; do
 if grep -q Traceback logs/v21_quality.log; then echo QUALITY_FAILED; exit 1; fi
 sleep 10
done
while pgrep -u "$USER" -f '^[^ ]*/python( -u)? scripts/run_layerkv_v21_suite.py scripts/v21_quality_jobs.json$' >/dev/null; do sleep 5; done
mkdir -p profiles/v21
for variant in original split_shared hybrid_identity hybrid_shared; do
 mode=shared
 args=(--diagnostic --nvtx)
 if [ "$variant" = original ]; then
  mode=original
 elif [ "$variant" = hybrid_identity ]; then
  mode=identity
  args+=(--layout hybrid --metadata-cache)
 elif [ "$variant" = hybrid_shared ]; then
  args+=(--layout hybrid --metadata-cache --packed-ops)
 fi
 if [ "$mode" = shared ]; then args+=(--adapters assets/v21_refit27_s200); fi
 "$NSYS" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o profiles/v21/nsys_$variant "$PY" scripts/profile_layerkv_v21.py --profiler nsys --profile-output profiles/v21/nsys_${variant}_meta -- --mode "$mode" "${args[@]}" --stats profiles/v21/nsys_$variant.extra.jsonl -- --task trec --budget 25 --period 8 --samples 8 --warmup 4 --bundle-dir data/sharedkv_splits/test --output profiles/v21/nsys_${variant}_run
 "$NSYS" export --type sqlite --output profiles/v21/nsys_$variant.sqlite profiles/v21/nsys_$variant.nsys-rep
done
"$PY" scripts/profile_layerkv_v21.py --profiler viztracer --profile-output profiles/v21/viz_hybrid_shared -- --mode shared --diagnostic --nvtx --layout hybrid --metadata-cache --packed-ops --adapters assets/v21_refit27_s200 -- --task trec --budget 25 --period 8 --samples 8 --warmup 4 --bundle-dir data/sharedkv_splits/test --output profiles/v21/viz_hybrid_shared_run
echo PROFILING_COMPLETE
"$PY" scripts/run_layerkv_v21_suite.py scripts/v21_holdout_gpu2_jobs.json
"$PY" scripts/run_layerkv_v21_suite.py scripts/v21_fixed_jobs.json
echo GPU2_QUALITY_COMPLETE
