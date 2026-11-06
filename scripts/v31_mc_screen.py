"""v31 many-class many-shot screen (Banking77 / CLINC150 / TREC-fine in slots sst2 / subj / trec).
build : prefix store + chunk-16 Pcache + K4 selector index (same code path as v29_quality_pipeline.py build).
screen: dev500 only, no readers: NS-P4 at 5 / 10 / 14.2857 / 25 / 50 % and Full KV (identity, budget 100).
Pre-declared criterion (written to data/v31_mc/screen_rule.json BEFORE any screen run): a dataset is budget-sensitive if
acc(NS-P4, 25%) - acc(NS-P4, 14.2857%) >= 3.0 pp on dev500. The screen only decides which datasets get the full pipeline;
all screened datasets are reported.
Usage: v31_mc_screen.py build <gpu> | screen <gpu,gpu,...>"""
from pathlib import Path
import json, os, runpy, subprocess, sys, time
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY, MODEL = C['R'], C['PY'], C['MODEL']; sys.path.insert(0, str(R / 'src'))
SLOTS = ['sst2', 'subj', 'trec']; NAMES = {'sst2': 'banking77', 'subj': 'clinc150', 'trec': 'trecfine'}
# the serving CLI registers --store-tasks sst2,subj,trec,rte, so the store also holds a 4th slot: the original RTE prefix
# (copied from the frozen v28 RTE bundle; registration only, never evaluated here)
STORE_SLOTS = SLOTS + ['rte']
D = R / 'data/v31_mc'; STORE = R / 'runtime_data/v31_mc_store'; PCACHE = R / 'runtime_data/v31_mc_pcache'; INDEX = R / 'assets/v31_mc_index'
W = R / 'studies/v31/mc_screen'; W.mkdir(parents=True, exist_ok=True)
step = sys.argv[1]
if step == 'build':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    from contiguous_fuxian.sparse_qwen_reprefill import prepare_prefix_store
    from contiguous_fuxian.flexgen_pcache import FlexGenPcacheConfig, FlexGenPcacheStore
    import torch
    t0 = time.time(); marker = PCACHE / '.contiguous_fuxian_complete'
    filler = D / 'eval' / 'rte.jsonl'
    if not filler.exists():
        filler.write_text((R / 'studies/v28_frozen/bundles/rte_b25/eval/rte.jsonl').read_text().splitlines()[0] + chr(10))
    if not marker.exists():
        assert not STORE.exists() and not PCACHE.exists()
        info = prepare_prefix_store(model_path=MODEL, bundle_dir=D / 'eval', store_root=STORE, tasks=STORE_SLOTS, samples_per_task=1, device='cuda', dtype='bfloat16')
        torch.cuda.empty_cache(); t1 = time.time()
        pc = FlexGenPcacheStore(FlexGenPcacheConfig(flexgen_root=R / 'vendor/flexgen', kv_dir=PCACHE, chunk_size=16, gpu_cache_mb=0, cpu_cache_mb=0,
                                                    cache_type='LRU', selector_kv_head_ids=(0, 1, 2, 3)))
        for t in STORE_SLOTS: pc.add_task(store_root=STORE, task=t)
        for p in PCACHE.iterdir():
            if p.is_file():
                with p.open('rb') as f: os.fsync(f.fileno())
        marker.write_text(json.dumps(dict(schema_version=3, method='v31_mc_prepared_prefix_native_full_kv', chunk_size=16,
            selector_kv_head_ids=[0, 1, 2, 3], online_selection=True, physical_layout='plain-logical-token-order',
            impress_reorder_sha256=None, registered_store_tasks=STORE_SLOTS, build_report='studies/v31/mc_screen/build.json'), indent=2))
        t2 = time.time()
    else: t1 = t2 = t0; info = {}
    subprocess.run([PY, '-u', 'scripts/build_selector_index.py', '--source-pcache-dir', str(PCACHE), '--store-root', str(STORE),
                    '--output-dir', str(INDEX), '--tasks', ','.join(STORE_SLOTS), '--source-task-order', ','.join(STORE_SLOTS), '--model-path', MODEL],
                   cwd=R, check=True, env=dict(os.environ, PYTHONPATH='src'))
    (W / 'build.json').write_text(json.dumps(dict(prefill_s=t1 - t0, pcache_s=t2 - t1, index_s=time.time() - t2,
        prefix={k: vars(v) for k, v in info.items()}), indent=1, default=str))
    print('BUILD_DONE', flush=True)
else:
    rule = D / 'screen_rule.json'
    if not rule.exists():
        rule.write_text(json.dumps(dict(written_unix=time.time(), metric='acc(NS-P4,25%) - acc(NS-P4,14.2857%) on dev500, percentage points',
                                        threshold_pp=3.0, decision='>= threshold: run the full v30 pipeline (P4 readers, dev selection, test, '
                                        'CAKE/LAVa, timing); otherwise report screen results only', datasets=NAMES), indent=1))
    gpus = sys.argv[2].split(','); jobs = []
    for slot in SLOTS:
        for b in (5, 10, 100 * 16 / 28 / 4, 25, 50):
            jobs.append((f'{NAMES[slot]}_NSP4_k{b:.4f}'.rstrip('0').rstrip('.').replace('.', 'p'), dict(task=slot, budget=b, period=4), float(b) != int(b)))
        jobs.append((f'{NAMES[slot]}_FULLKV', dict(task=slot, budget=100, period=8), False))
    queues = {g: [] for g in gpus}
    for i, (name, spec, ext) in enumerate(jobs):
        g = gpus[i % len(gpus)]
        queues[g].append(dict(name=name, argv=[PY, C['entry'](ext), *C['chain'](W / 'runs' / name, samples=500, bundle=D / 'dev', pcache=PCACHE,
                                                                               store=STORE, index=INDEX, **spec)]))
    C['run_queues'](W, queues)
    res = {}
    for name, spec, _ in jobs:
        rows, t = C['load_run'](W / 'runs' / name)
        res[name] = dict(acc=100 * t['accuracy'], MB=t['mean_physical_prefetch_kv_bytes'] / 1e6)
    r = json.loads(rule.read_text()); verdict = {}
    for slot in SLOTS:
        n = NAMES[slot]; a25, a14 = res[f'{n}_NSP4_k25']['acc'], res[f'{n}_NSP4_k14p2857']['acc']
        verdict[n] = dict(drop_pp=a25 - a14, sensitive=a25 - a14 >= r['threshold_pp'])
    (W / 'screen_result.json').write_text(json.dumps(dict(rule=r, runs=res, verdict=verdict), indent=1))
    print(json.dumps(dict(runs=res, verdict=verdict), indent=1)); print('SCREEN_DONE', flush=True)
