"""v32 (2): sensitivity-ordered revert for RAPS-P4 at 25 % (Qwen2.5-7B, the 4 classification tasks). Reader weights unchanged.
Phase fit : on 64 held-out unlabeled train-pool queries (data/v32/sens64), run NS-P4 (teacher), the full
            top graph (G16; SST-2: G21) and, for every consumer l, the graph with only l reverted to native KV.
            D(config) = mean over queries of KL(p_NS || p_config) on the label distribution (label-free, as the v29 rule).
            priority_l = D(top) - D(top without l); revert order = descending priority (ties: ascending layer).
            Per-consumer extra bytes are one layer bank each (equal exact per-layer budgets), so no byte normalisation.
Phase dev : G20s / G24s = revert the first 4 / 8 consumers of the new order (SST-2: G24s = first 3), dev500;
            frozen rule rev 3 over {I, G24s, G20s, top} (NS-P4 / top dev runs reused from studies/v30/quality).
Phase test: G24s / G20s on the frozen test sets (paired with the existing NS-P4 / RAPS-P4 test runs).
Usage: v32_layer_sens.py fit|dev|test <gpu,gpu,...>  |  report"""
from pathlib import Path
import json, math, runpy, sys
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY = C['R'], C['PY']; TASKS = C['TASKS']; sys.path.insert(0, str(R / 'src'))
W = R / 'studies/v32/layer_sens'; W.mkdir(parents=True, exist_ok=True)
Q = R / 'studies/v30/quality/runs'
P4R = {t: R / f'studies/v30/p4_readers/{t}/readers' for t in ('sst2', 'trec', 'rte')}; P4R['subj'] = R / 'studies/v29/subj_p4/readers'
TOP = {t: ('G21' if t == 'sst2' else 'G16') for t in TASKS}
# sensitivity queries: data/v32/sens64 (64 new train-pool queries per task, disjoint from reader fit/val, dev and test;
# the runtime rejects reader fit/val queries as leakage) - see scripts/prepare_v32_sens64.py
FIT = {t: R / 'data/v32/sens64' for t in TASKS}
step = sys.argv[1]

def consumers(t):
    import torch
    b = torch.load(P4R[t] / TOP[t] / f'{t}.pt', map_location='cpu', weights_only=False)
    return b, [l for l in range(28) if b['sources'][l] != l]

def dist(r):
    s = r['label_candidate_mean_logprobs']; ks = sorted(s); m = max(s.values()); e = [math.exp(s[k] - m) for k in ks]; z = sum(e)
    return [x / z for x in e]
def kl(p, q): return sum(a * (math.log(a) - math.log(max(b, 1e-30))) for a, b in zip(p, q) if a > 0)
def D(base, other):
    if set(base) != set(other): raise ValueError('UID sets differ')
    return sum(kl(dist(base[u]), dist(other[u])) for u in base) / len(base)

def run(jobs, gpus):
    queues = {g: [] for g in gpus}
    for i, (name, spec) in enumerate(jobs):
        g = gpus[i % len(gpus)]
        queues[g].append(dict(name=name, argv=[PY, C['entry'](False), *C['chain'](W / 'runs' / name, budget=25, period=4, pcache=C['PCACHE'][g], **spec)]))
    C['run_queues'](W, queues)

def derive(t, name, revert):
    import torch
    from contiguous_fuxian.sharedkv_v21.bundles import isolate
    b, cons = consumers(t); d = W / 'adapters' / t / name; p = d / f'{t}.pt'
    if not p.exists():
        d.mkdir(parents=True, exist_ok=True); child = isolate(b, sorted(set(cons) - set(revert)))
        with p.open('xb') as f: torch.save(child, f)
    return d

if step == 'fit':
    gpus = sys.argv[2].split(','); jobs = []
    for t in TASKS:
        n = sum(1 for l in (FIT[t] / f'{t}.jsonl').read_text().splitlines() if l.strip())
        fs = dict(task=t, samples=n, bundle=FIT[t])
        jobs.append((f'fit_{t}_NS', fs)); jobs.append((f'fit_{t}_{TOP[t]}', dict(fs, adapters=P4R[t] / TOP[t])))
        for l in consumers(t)[1]: jobs.append((f'fit_{t}_rev{l:02d}', dict(fs, adapters=derive(t, f'rev{l:02d}', [l]))))
    run(jobs, gpus)
    order = {}
    for t in TASKS:
        ns, _ = C['load_run'](W / f'runs/fit_{t}_NS'); top, _ = C['load_run'](W / f'runs/fit_{t}_{TOP[t]}'); d0 = D(ns, top); pr = {}
        for l in consumers(t)[1]:
            r, _ = C['load_run'](W / f'runs/fit_{t}_rev{l:02d}'); pr[l] = d0 - D(ns, r)
        b, cons = consumers(t); old = sorted(cons, key=lambda l: (-abs(l - b['sources'][l]), l))
        order[t] = dict(D_top=d0, priority=pr, new_order=sorted(cons, key=lambda l: (-pr[l], l)), old_order=old)
    (W / 'order.json').write_text(json.dumps(order, indent=1)); print(json.dumps({t: (o['new_order'], o['old_order']) for t, o in order.items()}), flush=True)
    print('FIT_DONE', flush=True)
elif step in ('dev', 'test'):
    gpus = sys.argv[2].split(','); order = json.loads((W / 'order.json').read_text()); jobs = []
    for t in TASKS:
        o = order[t]['new_order']; n = len(o)
        cands = [('G24s', o[:n - 4]), ('G20s', o[:n - 8])] if n >= 12 else [('G24s', o[:n - 4])]
        bundle = C['DEV'] if step == 'dev' else C['test_bundle'](t)
        for name, rev in cands:
            jobs.append((f'{step}_{t}_{name}', dict(task=t, samples=500 if step == 'dev' else C['N_TEST'][t], bundle=bundle, adapters=derive(t, name, rev))))
    run(jobs, gpus); print(step.upper() + '_DONE', flush=True)
else:   # report
    rule_tol = lambda n: math.ceil(0.01 * n)
    out = {}
    for t in TASKS:
        rows = {}
        for split, nsrun, toprun, oldfmt in (('dev', Q / f'ns_p4_dev_{t}_k25', Q / f'p4_dev_{TOP[t]}_{t}_k25', 'p4_dev_{g}_{t}_k25'),
                                             ('test', Q / f'ns_p4_test_{t}_k25', Q / f'p4_test_{TOP[t]}_{t}_k25', 'p4_test_{g}_{t}_k25')):
            ns, nst = C['load_run'](nsrun); res = {}
            cand = {'I': (ns, nst), TOP[t]: C['load_run'](toprun)}
            for g in ('G24', 'G20'):
                p = Q / oldfmt.format(g=g, t=t)
                if (p / 'summary.json').exists(): cand[g] = C['load_run'](p)
            for g in ('G24s', 'G20s'):
                p = W / f'runs/{split}_{t}_{g}'
                if (p / 'summary.json').exists(): cand[g] = C['load_run'](p)
            for g, (r, tt) in cand.items():
                pr = C['paired'](ns, r); res[g] = dict(acc=100 * pr['acc'], net=pr['net'], harmed=pr['harmed'], rescued=pr['rescued'],
                                                       bytes_vs_I=tt['mean_physical_prefetch_kv_bytes'] / nst['mean_physical_prefetch_kv_bytes'] - 1)
            rows[split] = res
        # frozen rule on dev, new candidate set {I, G24s, G20s, top}
        dev = rows['dev']; n = 500; tol = rule_tol(n); names = [g for g in ('I', 'G24s', 'G20s', TOP[t]) if g in dev]
        elig = [g for g in names if g == 'I' or dev[g]['net'] >= -tol]; bmin = min(dev[g]['bytes_vs_I'] for g in elig)
        near = [g for g in elig if (dev[g]['bytes_vs_I'] - bmin) / (1 + bmin) < 0.005]
        shared = lambda g: 0 if g == 'I' else 28 - int(g[1:3])
        pick = sorted(near, key=lambda g: (-dev[g]['net'], shared(g), names.index(g)))[0]
        out[t] = dict(rows=rows, selected_new=pick, eligible_new=elig)
    (R / 'results/v32').mkdir(parents=True, exist_ok=True)
    (R / 'results/v32/layer_sens.json').write_text(json.dumps(out, indent=1))
    for t, o in out.items():
        print(t, 'selected(new order):', o['selected_new'], 'eligible:', o['eligible_new'])
        for split in ('dev', 'test'):
            print('  ', split, {g: (round(v['acc'], 2), v['net'], f"{100 * v['bytes_vs_I']:+.1f}%") for g, v in o['rows'][split].items()})
