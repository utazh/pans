"""v29 QuALITY pipeline for one article set (pilot / f0 / f1 / f2), budget 25 by default.
build   : prefix store + chunk-16 Pcache + K4 selector index for the 4 articles (slots sst2/subj/trec/rte).
readers : per slot (one GPU each) and calibration source (syn = document-conditioned synthetic MCQs [main],
          ctl = questions from other articles [control]): teacher capture fit/val -> G16 plan (v26 geometry rule)
          -> R4 training (v26 recipe: 200 steps, eval 25, seed 24) -> frozen R0 -> G20/G24 via the frozen revert rule.
eval    : real questions of each article: I, G24, G20, G16 (syn), G16 (ctl), Full-KV (budget 100), and a no-document
          reference (question + options only, same label scoring). Reported as curves; no selection on real questions.
Usage: v29_quality_pipeline.py <set> build|readers|eval|nodoc [--budget 25]"""
from pathlib import Path
import copy, json, os, subprocess, sys, threading, time
R = Path('/home/shensicheng/qyd/pans/src'); PY = '/home/shensicheng/qyd/pans/venv/bin/python'
MODEL = '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct'; SLOTS = ['sst2', 'subj', 'trec', 'rte']
SET, STEP = sys.argv[1], sys.argv[2]
B = int(sys.argv[sys.argv.index('--budget') + 1]) if '--budget' in sys.argv else 25
# --steps N : training-budget diagnostic (default 200 = v26 recipe); reader dirs/tags get suffix _s<N> when N != 200.
# --sources a,b : calibration sources to train (default syn,ctl).  --arms a,b : eval arms override.
STEPS = int(sys.argv[sys.argv.index('--steps') + 1]) if '--steps' in sys.argv else 200
SFX = '' if STEPS == 200 else f'_s{STEPS}'
SOURCES = sys.argv[sys.argv.index('--sources') + 1].split(',') if '--sources' in sys.argv else ['syn', 'ctl']
ARMS = sys.argv[sys.argv.index('--arms') + 1].split(',') if '--arms' in sys.argv else None
D = R / f'data/v29/quality/{SET}'
STORE = R / f'runtime_data/v29_quality_{SET}_store'; PCACHE = R / f'runtime_data/v29_quality_{SET}_pcache'
INDEX = R / f'assets/v29_quality_{SET}_index'; W = R / f'studies/v29/quality/{SET}'
GPU = {'sst2': ('0', '0-7,32-39'), 'subj': ('1', '8-15,40-47'), 'trec': ('2', '28-31,60-63'), 'rte': ('3', '48-59')}
sys.path.insert(0, str(R / 'src'))

def sh(name, argv, gpu, cores, log_dir):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24')
    log_dir.mkdir(parents=True, exist_ok=True); t0 = time.time()
    with (log_dir / f'{name}.log').open('x') as f:
        rc = subprocess.run(['taskset', '-c', cores] + argv, cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    s = dict(name=name, gpu=gpu, returncode=rc, seconds=round(time.time() - t0, 1))
    with (log_dir / 'status.jsonl').open('a') as f: f.write(json.dumps(s) + '\n')
    print(json.dumps(s), flush=True)
    if rc: raise RuntimeError(name)

def chain(slot, mode, out, bundle, samples, budget, adapters=None, extra_v21=(), lead=('scripts/run_evidence_v26.py', '--admission', 'reuse-prefetch')):
    m = ['--mode', mode, '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{out}.extra.jsonl', *extra_v21]
    if adapters: m += ['--adapters', str(adapters), '--allow-candidate', '--packed-ops']
    return [PY, '-u', *lead, '--audit-jsonl', f'{out}.v26.jsonl', '--', '--model-path', MODEL, '--store-root', str(STORE),
            '--selector-index-dir', str(INDEX), '--', '--pcache-root', str(PCACHE), '--', *m, '--',
            '--task', slot, '--budget', str(budget), '--period', '8', '--max-tokens', '1', '--warmup', '0', '--samples', str(samples),
            '--profile', 'none', '--bundle-dir', str(bundle), '--output', str(out)]

def geometry_graph(row):  # identical to scripts/v26_training_worker.py graph(): geometry only, no labels/scores
    sources = list(range(28))
    old = json.loads((R / 'configs/v25_explore/g16.json').read_text())['sources']
    if all(row['selection_by_layer'][i] == row['selection_by_layer'][s] for i, s in enumerate(old)): return old
    need = 12
    for start in [24, 16, 8, 0]:
        remaining = list(range(start + 1, min(start + 8, 28))); groups = {}
        for i in remaining: groups.setdefault(tuple(row['selection_by_layer'][i]), []).append(i)
        for group in sorted(groups.values(), key=lambda g: (-len(g), -max(g))):
            if need <= 0: break
            n = min(need, len(group) - 1)
            if n <= 0: continue
            chosen = group[-(n + 1):]; src = chosen[len(chosen) // 2]
            for i in chosen: sources[i] = src
            need -= n
    assert need == 0
    return sources

def n_rows(p): return sum(1 for l in Path(p).read_text().splitlines() if l.strip())

if STEP == 'build':
    from contiguous_fuxian.sparse_qwen_reprefill import prepare_prefix_store
    from contiguous_fuxian.flexgen_pcache import FlexGenPcacheConfig, FlexGenPcacheStore
    import torch
    t0 = time.time(); info = {}
    marker = PCACHE / '.contiguous_fuxian_complete'
    reused = STORE.exists() and PCACHE.exists() and not marker.exists()
    if reused:  # a previous attempt finished store + all add_task calls and failed later (index step); reuse it
        assert sorted(p.name for p in STORE.iterdir() if p.is_dir()) == sorted(SLOTS)
        print('REUSING store/pcache from a previous attempt that completed them', flush=True)
    if not marker.exists() and not reused:  # store + pcache built once, then the marker is written (v26 p1 schema)
        assert not STORE.exists() and not PCACHE.exists()
        info = prepare_prefix_store(model_path=MODEL, bundle_dir=D / 'eval', store_root=STORE, tasks=SLOTS, samples_per_task=1, device='cuda', dtype='bfloat16')
        torch.cuda.empty_cache()
        pc = FlexGenPcacheStore(FlexGenPcacheConfig(flexgen_root=R / 'vendor/flexgen', kv_dir=PCACHE, chunk_size=16, gpu_cache_mb=0, cpu_cache_mb=0,
                                                    cache_type='LRU', selector_kv_head_ids=(0, 1, 2, 3)))
        for t in SLOTS: pc.add_task(store_root=STORE, task=t)
    for p in PCACHE.iterdir():
        if p.is_file():
            with p.open('rb') as f: os.fsync(f.fileno())
    if not marker.exists():
        marker.write_text(json.dumps(dict(schema_version=3, method='v29_quality_prepared_prefix_native_full_kv', chunk_size=16,
            selector_kv_head_ids=[0, 1, 2, 3], online_selection=True, physical_layout='plain-logical-token-order',
            impress_reorder_sha256=None, registered_store_tasks=SLOTS, build_report=f'studies/v29/quality/{SET}/build.json'), indent=2))
    t1 = t2 = time.time()
    subprocess.run([PY, '-u', 'scripts/build_selector_index.py', '--source-pcache-dir', str(PCACHE), '--store-root', str(STORE),
                    '--output-dir', str(INDEX), '--tasks', ','.join(SLOTS), '--source-task-order', ','.join(SLOTS), '--model-path', MODEL],
                   cwd=R, check=True, env=dict(os.environ, PYTHONPATH='src'))
    W.mkdir(parents=True, exist_ok=True)
    (W / 'build.json').write_text(json.dumps(dict(prefill_s=t1 - t0, pcache_s=t2 - t1, index_s=time.time() - t2,
        prefix={k: vars(v) for k, v in info.items()}), indent=1, default=str))
    print('BUILD_DONE', flush=True)

elif STEP == 'readers':
    from contiguous_fuxian.sharedkv_v24.records import load, FIELDS, PLAN_FORMAT, assert_plan
    from contiguous_fuxian.sharedkv_v21.bundles import isolate
    import torch
    logs = W / 'logs'
    def per_slot(slot):
        g, cores = GPU[slot]
        for src in SOURCES:
            base_tag = f'v29q_{SET}_{slot}_b{B}_{src}'; tag = base_tag + SFX; td = {}
            for split, role in (('fit', 'fit'), ('val', 'validation')):
                rec = R / f'runtime_data/{base_tag}_teacher_{split}'; res = W / 'capture' / f'{base_tag}_{split}'; res.parent.mkdir(parents=True, exist_ok=True)
                bundle = D / f'{src}_{split}'
                if rec.exists():  # reuse the identical teacher records (same data for every training budget)
                    td[split] = rec; continue
                sh(f'{tag}_capture_{split}', [PY, '-u', 'scripts/collect_teacher_v26.py', '--teacher-records', str(rec), '--split', role, '--',
                   *chain(slot, 'identity', res, bundle, n_rows(bundle / f'{slot}.jsonl'), B, extra_v21=('--diagnostic',),
                          lead=('--admission', 'original'))[2:]], g, cores, logs)
                td[split] = rec
            fit, val = load(td['fit'], slot, 'fit'), load(td['val'], slot, 'validation'); row = next(iter(fit.values()))
            sources = geometry_graph(row)
            for item in list(fit.values()) + list(val.values()): assert_plan(item, sources)
            plan = dict(format=PLAN_FORMAT, **{k: row[k] for k in FIELDS}, sources=sources, unique_sources=len(set(sources)),
                        design_query_hashes=sorted(fit), status='v29_quality_fixed_G16_geometry_only', quality_certified=False)
            plan_p = R / f'configs/{tag}_plan.json'
            with plan_p.open('x') as f: json.dump(plan, f, indent=2)
            ck = R / f'assets/{tag}/{slot}.pt'
            sh(f'{tag}_train', [PY, '-u', 'scripts/train_instrumented_v25.py', '--native-blocks', '4', '--worklog', str(W / f'{tag}_compute.json'), '--',
               '--model-path', MODEL, '--store-root', str(STORE), '--task', slot, '--teacher-fit', str(td['fit']), '--teacher-val', str(td['val']),
               '--plan', str(plan_p), '--steps', str(STEPS), '--eval-every', '25', '--seed', '24', '--device', 'cuda', '--output', str(ck)], g, cores, logs)
            r0 = W / 'readers' / f'{src}{SFX}_G16' / f'{slot}.pt'; r0.parent.mkdir(parents=True, exist_ok=True)
            from contiguous_fuxian.sharedkv_v27.frozen_control import export_frozen_r0
            export_frozen_r0(ck, r0)
            if src == 'syn':  # G20/G24 by the frozen label-free revert rule (rule rev 1-3)
                bundle = torch.load(r0, map_location='cpu', weights_only=False); s = bundle['sources']
                cons = [l for l in range(28) if s[l] != l]; order = sorted(cons, key=lambda l: (-abs(l - s[l]), l))
                for name, k in (('G20', 4), ('G24', 8)):
                    child = isolate(bundle, sorted(set(cons) - set(order[:k])))
                    p = W / 'readers' / f'syn{SFX}_{name}' / f'{slot}.pt'; p.parent.mkdir(parents=True, exist_ok=True)
                    with p.open('xb') as f: torch.save(child, f)
    errs = []
    def guarded(s):
        try: per_slot(s)
        except Exception as e: errs.append(f'{s}: {e!r}'); print('FAILED', s, repr(e), flush=True)
    ts = [threading.Thread(target=guarded, args=(s,)) for s in SLOTS]; [t.start() for t in ts]; [t.join() for t in ts]
    (W / f'readers_done{SFX}.json').write_text(json.dumps(dict(errors=errs, steps=STEPS, sources=SOURCES))); print('READERS_DONE', errs, flush=True)

elif STEP == 'eval':
    logs = W / 'eval_logs'
    def per_slot(slot):
        g, cores = GPU[slot]; bundle = D / 'eval'; n = n_rows(bundle / f'{slot}.jsonl')
        names = ARMS or ['I', 'FULLKV', 'syn_G24', 'syn_G20', 'syn_G16', 'ctl_G16']
        arms = [(a, 'identity', None, 100 if a == 'FULLKV' else B) if a in ('I', 'FULLKV') else (a, 'shared', W / 'readers' / a, B)
                for a in names]
        for name, mode, ad, bud in arms:
            out = W / 'eval' / f'{name}_{slot}'; out.parent.mkdir(parents=True, exist_ok=True)
            sh(f'eval_{name}_{slot}', chain(slot, mode, out, bundle, n, bud, adapters=ad), g, cores, logs)
    ts = [threading.Thread(target=per_slot, args=(s,)) for s in SLOTS]; [t.start() for t in ts]; [t.join() for t in ts]
    print('EVAL_DONE', flush=True)

elif STEP == 'nodoc':  # question + options only; same template minus the article; mean label-token log-prob
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    res = {}
    for slot in SLOTS:
        rows = [json.loads(l) for l in (D / 'eval' / f'{slot}.jsonl').read_text().splitlines() if l.strip()]; ok = 0
        for r in rows:
            prompt = '<|im_start|>user\nAnswer the multiple-choice question.\n\n' + r['query_text']
            ids = tok(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
            scores = {}
            for L in r['labels']:
                lid = tok(L, add_special_tokens=False).input_ids
                with torch.no_grad():
                    lp = torch.log_softmax(model(torch.cat([ids, torch.tensor([lid]).cuda()], 1)).logits[0].float(), -1)
                scores[L] = sum(float(lp[ids.shape[1] - 1 + j, t]) for j, t in enumerate(lid)) / len(lid)
            ok += max(scores, key=scores.get) == r['answer']
        res[slot] = dict(n=len(rows), acc=ok / len(rows)); print(slot, res[slot], flush=True)
    W.mkdir(parents=True, exist_ok=True); (W / 'nodoc.json').write_text(json.dumps(res, indent=1))
