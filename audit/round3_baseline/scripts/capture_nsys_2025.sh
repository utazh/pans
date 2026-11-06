#!/bin/bash
set -u
cd /home/panzihang/src/prism_pans
NSYS="$PWD/runtime_tools/nsight-2025.5.1/opt/nvidia/nsight-systems-cli/2025.5.1/target-linux-x64/nsys"
"$NSYS" --version
env CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 "$NSYS" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o profiles/trec_p1_nsys_2025 /home/panzihang/venvs/vllm-stable/bin/python scripts/run_round1.py --task trec --samples 4 --warmup 4 --profile nsys --output profiles/trec_p1_nsys_2025_run
code=$?
printf '%s\n' "$code" > profiles/nsys_2025_exitcode.txt
exit "$code"
