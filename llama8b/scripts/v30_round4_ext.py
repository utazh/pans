#!/usr/bin/env python3
"""v30 innermost runner: identical serving arguments to the guarded scripts/run_round4.py (optimized variant,
profile none), plus three additive options used only by the v30 study:
  --budget FLOAT                 any keep ratio in (0, 100] percent (plan written to configs/v30_plans/);
  --layer-budget-profile PATH    non-uniform per-layer block budgets (engine's existing layer_budget support);
  --static-selection PATH        one fixed per-layer token selection per task prefix (query-independent),
  --oracle-selection-dir DIR     one fixed per-layer token selection per query (keyed by sharedkv.io.query_hash).
Fixed selections reuse sharedkv.selection.fixed_selection (the v24 replay hook): the online selector still runs and
is charged; only the selected IDs are replaced. Reached through scripts/v30_entry.py (runpy redirect)."""
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--bundle-dir', type=Path, required=True)
p.add_argument('--max-tokens', type=int, default=1)
p.add_argument('--task', choices=['sst2', 'subj', 'trec', 'rte'], required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--samples', type=int, default=1000000)
p.add_argument('--warmup', type=int, default=32)
p.add_argument('--profile', choices=['none'], default='none')
p.add_argument('--period', type=int, choices=[1, 2, 4, 8], default=8)
p.add_argument('--timing', choices=['deferred', 'sync'], default='deferred')
p.add_argument('--budget', type=float, required=True)
p.add_argument('--variant', choices=['optimized'], default='optimized')
p.add_argument('--layer-budget-profile', type=Path)
g = p.add_mutually_exclusive_group()
g.add_argument('--static-selection', type=Path)
g.add_argument('--oracle-selection-dir', type=Path)
a = p.parse_args()
if not 0 < a.budget <= 100: raise SystemExit('budget must be in (0, 100]')
a.output = a.output.resolve(); a.output.relative_to(ROOT)
if a.output.exists(): raise SystemExit(f'output exists: {a.output}')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(ROOT / 'src'))
import contiguous_fuxian.flexgen_qwen_reprefill as runner
from contiguous_fuxian.sharedkv.selection import fixed_selection
from contiguous_fuxian.sharedkv.io import query_hash

ratio = a.budget / 100
tag = (f'{a.budget:.4f}' if round(a.budget, 4) == a.budget else f'{a.budget:.10f}').rstrip('0').rstrip('.').replace('.', 'p')   # 4-decimal budgets keep their v30 plan names
plan = ROOT / 'configs/v30_plans' / f'qwen25_k{tag}_uniform.json'
body = json.loads((ROOT / 'configs/round2/qwen25_k025_uniform.json').read_text())
body['metadata']['keep_ratio'] = ratio
text = json.dumps(body, indent=2) + '\n'
if plan.exists():
    if plan.read_text() != text: raise SystemExit(f'plan content mismatch: {plan}')
else:
    plan.parent.mkdir(parents=True, exist_ok=True)
    tmp = plan.with_suffix(f'.tmp{os.getpid()}'); tmp.write_text(text); os.replace(tmp, plan)

argv = [
    '--model-path', os.environ.get('PANS_MODEL_PATH', '/data1/llm/Qwen/Qwen2.5-7B-Instruct'),
    '--bundle-dir', str(a.bundle_dir),
    '--store-root', str(ROOT / 'runtime_data/prefix_store'),
    '--plan', str(plan),
    '--flexgen-root', str(ROOT / 'vendor/flexgen'), '--flexgen-kv-dir', str(ROOT / 'runtime_data/pcache_c16'),
    '--reuse-flexgen-kv', '--tasks', a.task, '--store-tasks', 'sst2,subj,trec,rte',
    '--output-dir', str(a.output), '--samples-per-task', str(a.samples), '--max-tokens', str(a.max_tokens),
    '--accuracy-scoring', 'label_continuation_loglikelihood', '--dtype', 'bfloat16', '--device', 'cuda',
    '--gpu-cache-mb', '55', '--cpu-cache-mb', '131', '--cache-type', 'CKLFU',
    '--online-selection', '--probe-query-heads', os.environ.get('PANS_PROBE_QUERY_HEADS', '0,7,14,21'),
    '--selector-kv-head-ids', os.environ.get('PANS_SELECTOR_KV_HEAD_IDS', '0,1,2,3'),
    '--impress-selection-block-size', '16', '--impress-selection-period-size', str(a.period),
    '--promixed-gqa-selection', '--promixed-reuse-mode', 'fixed', '--promixed-coverage-fraction', '0.5',
    '--no-promixed-adaptive-coverage', '--exact-layer-block-budget', '--impress-async-prefetch',
    '--impress-period-prefetch-size', '1', '--no-impress-rolling-period-prefetch', '--no-impress-priority-prefetch',
    '--impress-deferred-compute-timing' if a.timing == 'deferred' else '--no-impress-deferred-compute-timing',
    '--no-impress-value-ordered-prefetch', '--no-defer-cache-score-updates',
    '--impress-known-period-prefetch' if a.period > 1 else '--no-impress-known-period-prefetch',
    '--period-size', '8', '--subperiod-size', '4', '--expected-keep-ratio', str(ratio),
    '--warmup-passes', '1' if a.warmup else '0', '--warmup-samples-per-task', str(max(1, a.warmup)),
    '--selector-index-dir', str(ROOT / 'assets/selector_index_k4_g32')]
if a.layer_budget_profile:
    argv += ['--layer-budget-profile', str(a.layer_budget_profile)]

info = dict(budget_percent=a.budget, keep_ratio=ratio, plan=str(plan), period=a.period,
            layer_budget_profile=str(a.layer_budget_profile) if a.layer_budget_profile else None,
            static_selection=None, oracle_selection_dir=None)
if a.static_selection:
    sel = json.loads(a.static_selection.read_text())
    if sel['task'] != a.task: raise SystemExit('static selection task mismatch')
    by_layer = [list(map(int, ids)) for ids in sel['selected_by_layer']]
    info['static_selection'] = dict(path=str(a.static_selection), sha256=hashlib.sha256(a.static_selection.read_bytes()).hexdigest(),
                                    method=sel.get('method'), blocks=sel.get('blocks_total'))
    unfrozen = runner.greedy_flexgen_completion
    def frozen(*args, **kw):
        if kw['prefix_tokens'] != sel['prefix_tokens']: raise ValueError('static selection prefix length mismatch')
        with fixed_selection(kw['loader'], by_layer):
            return unfrozen(*args, **kw)
    runner.greedy_flexgen_completion = frozen
elif a.oracle_selection_dir:
    info['oracle_selection_dir'] = str(a.oracle_selection_dir)
    unfrozen = runner.greedy_flexgen_completion
    def frozen(*args, **kw):
        digest = query_hash(kw['query_token_ids'])
        sel = json.loads((a.oracle_selection_dir / f'{digest}.json').read_text())
        if sel['query_hash'] != digest or sel['task'] != a.task or kw['prefix_tokens'] != sel['prefix_tokens']:
            raise ValueError('oracle selection identity mismatch')
        with fixed_selection(kw['loader'], [list(map(int, ids)) for ids in sel['selected_by_layer']]):
            return unfrozen(*args, **kw)
    runner.greedy_flexgen_completion = frozen

sys.argv = [runner.__file__] + argv
runner.main()
if a.output.exists():
    (a.output / 'command.json').write_text(json.dumps({'args': argv, 'v30': info, 'script': __file__,
                                                       'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}, indent=2))
    sys.path.insert(0, str(ROOT / 'vendor/flexgen'))
    import my_pcache_fast
    (a.output / 'engineering_metrics.json').write_text(json.dumps(my_pcache_fast._STAGING_POOL.metrics(), indent=2))
