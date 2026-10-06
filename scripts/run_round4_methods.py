#!/usr/bin/env python3
"""v29 baseline innermost runner: ContiguousKV / IMPRESS reproductions with the canonical arguments of
scripts/run_prism_max_cell.sh (fp16 selector, no K4 index, CKLFU 55/131 MiB, P8/SubPeriod 4, no deferred
cache-score updates), on the SAME prefix store, bundles, label scoring and storage backend as our system.
Called through run_evidence_v26 -> v25 -> v23 (cold) -> v21 --mode original via scripts/v29_baseline_entry.py,
or directly (--build-kv) to materialize an IMPRESS reordered chunk-64 Pcache."""
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--method', choices=['contigkv', 'impress'], required=True)
p.add_argument('--bundle-dir', type=Path, required=True)
p.add_argument('--max-tokens', type=int, default=1)
p.add_argument('--task', choices=['sst2', 'subj', 'trec', 'rte'], required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--samples', type=int, default=1000000)
p.add_argument('--warmup', type=int, default=32)
p.add_argument('--budget', type=int, choices=[5, 10, 25, 50], required=True)
p.add_argument('--period', type=int, choices=[8], default=8)
p.add_argument('--profile', choices=['none'], default='none')
p.add_argument('--impress-reorder', type=Path)
p.add_argument('--kv-dir', type=Path, default=ROOT / 'runtime_data/pcache_c16', help='overridden by v23 --pcache-root')
p.add_argument('--build-kv', action='store_true', help='materialize a new Pcache instead of --reuse-flexgen-kv')
a = p.parse_args()
a.output = a.output.resolve(); a.output.relative_to(ROOT)
if a.output.exists(): raise SystemExit(f'output exists: {a.output}')
if a.method == 'impress' and not (a.impress_reorder and a.impress_reorder.is_file()):
    raise SystemExit('IMPRESS requires --impress-reorder manifest')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(ROOT / 'src'))
import contiguous_fuxian.flexgen_qwen_reprefill as runner
argv = ['--model-path', os.environ.get('PANS_MODEL_PATH', '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct'), '--bundle-dir', str(a.bundle_dir),
        '--store-root', str(ROOT / 'runtime_data/prefix_store'),
        '--plan', str(ROOT / f'configs/qwen25_k{a.budget:03d}_{a.method}.json'),
        '--output-dir', str(a.output), '--flexgen-root', str(ROOT / 'vendor/flexgen'), '--flexgen-kv-dir', str(a.kv_dir),
        '--tasks', a.task, '--store-tasks', 'sst2,subj,trec,rte', '--samples-per-task', str(a.samples),
        '--max-tokens', str(a.max_tokens), '--accuracy-scoring', 'label_continuation_loglikelihood',
        '--device', 'cuda', '--dtype', 'bfloat16', '--gpu-cache-mb', '55', '--cpu-cache-mb', '131', '--cache-type', 'CKLFU',
        '--prefetch-time-budget', '10000', '--online-selection', '--period-size', '8', '--subperiod-size', '4',
        '--expected-keep-ratio', str(a.budget / 100), '--warmup-passes', '1' if a.warmup else '0',
        '--warmup-samples-per-task', str(max(1, a.warmup)), '--no-defer-cache-score-updates']
if not a.build_kv: argv.append('--reuse-flexgen-kv')
if a.method == 'contigkv':
    argv += ['--probe-query-heads', '0,1,2', '--selector-kv-head-ids', os.environ.get('PANS_SELECTOR_KV_HEAD_IDS', '0,1,2,3'), '--similarity-alpha', '0.6']
else:
    argv += ['--probe-query-heads', '0,1,2', '--selector-kv-head-ids', '0', '--similarity-alpha', '0.6',
             '--impress-reorder-manifest', str(a.impress_reorder), '--no-impress-async-prefetch']
sys.argv = [runner.__file__] + argv
runner.main()
if a.output.exists():
    (a.output / 'command.json').write_text(json.dumps({'args': sys.argv[1:], 'method': a.method, 'script': __file__,
                                                       'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}, indent=2))
