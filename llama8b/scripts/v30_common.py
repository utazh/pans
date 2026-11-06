"""v30 shared helpers: argv builders for the unchanged evidence chain, a GPU job runner, and strict paired loading.
Used via runpy/import by the v30_* scripts. Nothing here changes serving code."""
from pathlib import Path
import hashlib, json, math, os, subprocess, threading, time
# second-model runs (Llama-3.1-8B worktree) override root / model via env; defaults = the Qwen2.5-7B tree
R = Path(os.environ.get('V30_ROOT', '/home/panzihang/src/prism_layerkv')); PY = '/home/panzihang/venvs/vllm-stable/bin/python'
MODEL = os.environ.get('V30_MODEL', '/data1/llm/Qwen/Qwen2.5-7B-Instruct')
DEV = R / 'data/v29/p0/dev'
N_TEST = {'sst2': 872, 'subj': 1003, 'trec': 500, 'rte': 277}
TASKS = ['sst2', 'subj', 'trec', 'rte']
# quality-only pcache copies (never the SSD timing copy runtime_data/v29_pcache_gpu2_ssd)
PCACHE = {'1': R / 'runtime_data/v27_pcache_gpu2', '2': R / 'runtime_data/v27_pcache_gpu3',
          '3': Path('/data1/pans/pans_kv/pcache_a2_gpu3'), '0': Path('/data1/pans/pans_kv/pcache_a2_gpu2')}
CORES = {'0': '32-39', '1': '40-47', '2': '48-55', '3': '56-59,12-15'}
if os.environ.get('V30_PCACHE'):   # one quality Pcache for every GPU (second-model tree)
    PCACHE = {g: Path(os.environ['V30_PCACHE']) for g in CORES}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def test_bundle(task, budget=25):
    """Frozen v28 test bundle; the eval split is identical across budgets of a task (checked by test_identity)."""
    return R / f'studies/v28_frozen/bundles/{task}_b{budget}/eval'

def chain(out, *, task, budget, period, samples, bundle, pcache, adapters=None, ext=(), warmup=0, cold=None, store=None, index=None):
    """run_evidence_v26 argument list (v26 -- v25 -- v23 -- v21 -- round4)."""
    m = ['--mode', 'identity' if adapters is None else 'shared', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid',
         '--stats', f'{out}.extra.jsonl']
    if adapters is not None: m += ['--adapters', str(adapters), '--allow-candidate', '--packed-ops']
    v23 = ['--pcache-root', str(pcache)]
    if cold: v23 += ['--cold', 'ssd-cold', '--read-advice', cold, '--owned-cache-root', str(pcache), '--cold-log', f'{out}.cold.jsonl']
    b = repr(float(budget)) if float(budget) != int(budget) else str(int(budget))   # full precision (profile means must match)
    return ['--admission', 'reuse-prefetch', '--audit-jsonl', f'{out}.v26.jsonl', '--',
            '--model-path', MODEL, '--store-root', str(store or R / 'runtime_data/prefix_store'),
            '--selector-index-dir', str(index or R / 'assets/selector_index_k4_g32'), '--', *v23, '--', *m, '--',
            '--task', task, '--budget', b, '--period', str(period), '--max-tokens', '1', '--warmup', str(warmup),
            '--samples', str(samples), '--profile', 'none', '--bundle-dir', str(bundle), '--output', str(out), *ext]

def entry(ext_needed):
    return str(R / ('scripts/v30_entry.py' if ext_needed else 'scripts/run_evidence_v26.py'))

def run_queues(out_dir, queues):
    """queues: {gpu: [dict(name, argv)]}; runs each queue serially, queues in parallel; skips jobs whose
    summary.json already exists (resumable); appends status.jsonl."""
    out_dir = Path(out_dir); (out_dir / 'logs').mkdir(parents=True, exist_ok=True); (out_dir / 'runs').mkdir(exist_ok=True)
    def run(g):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24')
        for j in queues[g]:
            run_dir = out_dir / 'runs' / j['name']
            if (run_dir / 'summary.json').exists(): continue
            side = list(out_dir.glob(f"runs/{j['name']}.*"))
            if run_dir.exists() or side:   # failed partial run (dir and/or sidecar files): move aside, never overwrite
                k = 0
                while (out_dir / 'runs' / f"_failed_{j['name']}_{k}").exists() or (out_dir / 'runs' / f"_failed_{k}_{j['name']}.v26.jsonl.effective_cli.json").exists(): k += 1
                if run_dir.exists(): run_dir.rename(out_dir / 'runs' / f"_failed_{j['name']}_{k}")
                for p in side: p.rename(p.with_name(f"_failed_{k}_" + p.name))
            t0 = time.time(); log = out_dir / 'logs' / f"{j['name']}.log"
            with log.open('a') as f:
                rc = subprocess.run(['taskset', '-c', CORES[g]] + j['argv'], cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            s = dict(name=j['name'], gpu=g, returncode=rc, seconds=round(time.time() - t0, 1), t=time.time())
            with (out_dir / 'status.jsonl').open('a') as f: f.write(json.dumps(s) + '\n')
            print(json.dumps(s), flush=True)
    ts = [threading.Thread(target=run, args=(g,)) for g in queues]; [t.start() for t in ts]; [t.join() for t in ts]

def load_run(run):
    """Strict loader: unique UIDs, finite bytes. Returns (rows by uid, summary task dict)."""
    run = Path(run); rows = {}
    for line in (run / 'scored_records.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['uid'] in rows: raise ValueError(f'{run}: duplicate uid {r["uid"]}')
        if not isinstance(r['correct'], bool): raise ValueError(f'{run}: non-bool correct')
        rows[r['uid']] = r
    t = list(json.loads((run / 'summary.json').read_text())['tasks'].values())[0]
    if t['samples'] != len(rows): raise ValueError(f'{run}: summary samples {t["samples"]} != rows {len(rows)}')
    b = t.get('mean_physical_prefetch_kv_bytes')
    if b is None or not math.isfinite(b): raise ValueError(f'{run}: bytes missing')
    return rows, t

def paired(base, other):
    """Paired comparison on identical query sets; raises on any mismatch in UID set, label, or query."""
    if set(base) != set(other): raise ValueError('paired runs have different UID sets')
    for u in base:
        for f in ('label', 'gold', 'query_text'):
            if f in base[u] and base[u].get(f) != other[u].get(f): raise ValueError(f'uid {u}: {f} differs')
    n = len(base); cb = sum(r['correct'] for r in base.values()); co = sum(other[u]['correct'] for u in base)
    harm = sum(base[u]['correct'] and not other[u]['correct'] for u in base)
    resc = sum(other[u]['correct'] and not base[u]['correct'] for u in base)
    dis = sum(base[u]['prediction'] != other[u]['prediction'] for u in base)
    return dict(n=n, acc_base=cb / n, acc=co / n, delta_pp=100 * (co - cb) / n, net=co - cb, harmed=harm, rescued=resc, disagree=dis)
