"""v30 report (read-only on runs). Writes results/v30/:
  p4_selection.json      frozen v29 rule (rev 3) applied to P4 dev runs: NS-P4 vs G24/G20/G16-P4 at 25 %
  quality_points.csv     every test-set point: task, family, label, budget, period, accuracy, bytes, paired stats vs NS-P8 same budget
  p4_vs_p8.csv           NS P4 vs P8 (dev and test) per task x budget with harmed / rescued / disagreements
  frontier.csv           each shared / baseline point vs the NS-P8 accuracy-bytes curve interpolated at the same bytes
  timing.csv (if studies/v30/timing exists)  per task x arm block-paired cold TTFT ratios vs NS25 and vs RAPS points
Usage: v30_report.py [select-only]"""
from pathlib import Path
import csv, json, math, runpy, statistics as st, sys
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R = C['R']; TASKS = C['TASKS']; load, paired = C['load_run'], C['paired']
OUT = R / 'results/v30'; OUT.mkdir(parents=True, exist_ok=True)
Q = R / 'studies/v30/quality/runs'; V29 = R / 'studies/v29'
EQ = [100 * 16 / 28 / 4, 100 * 20 / 28 / 4, 100 * 24 / 28 / 4]
tag = lambda b: f'{b:.4f}'.rstrip('0').rstrip('.').replace('.', 'p')
rule = json.loads((R / 'configs/v29_selection_rule.json').read_text()); assert rule['revision'] == 3
SHARED = lambda name: 0 if name == 'I' else 28 - int(name[1:])          # shared consumers of G<k>
P4R = {t: R / f'studies/v30/p4_readers/{t}/readers' for t in ('sst2', 'trec', 'rte')}; P4R['subj'] = R / ('studies/v30/p4_readers/subj/readers' if __import__('os').environ.get('V30_ROOT') else 'studies/v29/subj_p4/readers')
P4C = {t: sorted((d.name for d in P4R[t].iterdir() if (d / f'{t}.pt').exists()), key=lambda n: -int(n[1:])) if P4R[t].exists() else [] for t in TASKS}

def try_load(p):
    p = Path(p)
    return load(p) if (p / 'summary.json').exists() else None

# ---------- P4 selection (dev) ----------
sel = {}
for t in TASKS:
    ORDER = ['I'] + P4C[t]                         # fixed name order I, then fewer-sharing first (rule tie order)
    runs = {'I': try_load(Q / f'ns_p4_dev_{t}_k25')}
    for g in P4C[t]: runs[g] = try_load(Q / f'p4_dev_{g}_{t}_k25')
    if not P4C[t] or any(v is None for v in runs.values()): sel[f'{t}_b25'] = dict(status='pending'); continue
    I_rows, I_t = runs['I']; n = len(I_rows); tol = math.ceil(0.01 * n); cands = {}
    for name in ORDER:
        rows, tt = runs[name]; p = paired(I_rows, rows)
        cands[name] = dict(p, bytes=tt['mean_physical_prefetch_kv_bytes'], bytes_vs_I=tt['mean_physical_prefetch_kv_bytes'] / I_t['mean_physical_prefetch_kv_bytes'] - 1,
                           eligible=name == 'I' or p['net'] >= -tol, shared_consumers=SHARED(name))
    elig = [k for k in ORDER if cands[k]['eligible']]; bmin = min(cands[k]['bytes'] for k in elig)
    near = [k for k in elig if (cands[k]['bytes'] - bmin) / bmin < 0.005]
    pick = sorted(near, key=lambda k: (-cands[k]['net'], cands[k]['shared_consumers'], ORDER.index(k)))[0]
    sel[f'{t}_b25'] = dict(status='selected', selected=pick, tolerance_correct=tol, eligible=elig, candidate_set=P4C[t], candidates=cands)
(OUT / 'p4_selection.json').write_text(json.dumps(dict(rule='configs/v29_selection_rule.json rev 3 (unchanged), applied to P4 dev500',
                                                       rule_sha256=C['sha'](R / 'configs/v29_selection_rule.json'), cells=sel), indent=1))
print('P4 selection:', {k: v.get('selected', v['status']) for k, v in sel.items()})
if sys.argv[1:2] == ['select-only']: raise SystemExit

# ---------- test-set points ----------
final8 = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
def p8_cand(cand, cell):
    for root in (V29 / 'test_quality/runs', V29 / 'p0_test_g/runs'):
        if (root / f'test_{cand}_{cell}' / 'summary.json').exists(): return root / f'test_{cand}_{cell}'
pts = []
PREFIX_TOKENS = {t: json.loads((R / f'runtime_data/prefix_store/{t}/metadata.json').read_text())['prefix_tokens'] for t in TASKS}
def add(task, family, label, budget, period, run, **extra):
    try: r = try_load(run)
    except ValueError:
        if family != 'FullKV': raise
        s = list(json.loads((Path(run) / 'summary.json').read_text())['tasks'].values())[0]
        rows = {json.loads(l)['uid']: json.loads(l) for l in (Path(run) / 'scored_records.jsonl').read_text().splitlines()}
        # full prefix KV of all 28 layers, K+V, 4 KV heads x 128 dims, fp16 storage
        r = (rows, dict(s, mean_physical_prefetch_kv_bytes=PREFIX_TOKENS[task] * 28 * 2 * 4 * 128 * 2))
    if r is None: return
    pts.append(dict(task=task, family=family, label=label, budget=budget, period=period, run=str(run),
                    acc=100 * r[1]['accuracy'], MB=r[1]['mean_physical_prefetch_kv_bytes'] / 1e6, _rows=r[0], **extra))
for t in TASKS:
    add(t, 'FullKV', 'Full KV', 100, None, V29 / f'a0/runs/fullkv_{t}')
    for b in (5, 10, 25, 50):
        cell = f'{t}_b{b}'
        add(t, 'NS-P8', f'NS {b}%', b, 8, V29 / f'test_quality/runs/test_I_{cell}')
        add(t, 'NS-P4', f'NS-P4 {b}%', b, 4, Q / f'ns_p4_test_{t}_k{b}')
        for g in ('G16', 'G20', 'G24'):
            p = p8_cand(g, cell)
            if p: add(t, 'RAPS-P8', f'{g} {b}%', b, 8, p, selected=final8[cell]['config'] == g)
        add(t, 'CKV', f'CKV {b}%', b, 8, V29 / f'c_quality/runs/contigkv_{t}_b{b}')
        add(t, 'IMPRESS', f'IMPRESS {b}%', b, 8, V29 / f'c_quality/runs/impress_{t}_b{b}')
    for b in EQ:
        add(t, 'NS-P8', f'NS {b:.1f}%', b, 8, Q / f'ns_eq_{t}_k{tag(b)}_p8')
        add(t, 'NS-P4', f'NS-P4 {b:.1f}%', b, 4, Q / f'ns_eq_{t}_k{tag(b)}_p4')
    for g in P4C[t]:
        add(t, 'RAPS-P4', f'{g}-P4 25%', 25, 4, Q / f'p4_test_{g}_{t}_k25', selected=sel.get(f'{t}_b25', {}).get('selected') == g)
    for m, M in (('cake', 'CAKE'), ('lava', 'LAVa')):
        for b in (10, *EQ, 25, 50): add(t, f'{M}-prefix', f'{M} {b:g}%', b, 8, Q / f'{m}_static_{t}_k{tag(b)}')
        for b in (EQ[0], 25): add(t, f'{M}-budget', f'{M}-budget {b:.1f}%', b, 8, Q / f'{m}_budget_{t}_k{tag(b)}')
        add(t, f'{M}-oracle', f'{M}-oracle 25%', 25, 8, Q / f'{m}_oracle_{t}_k25')

# paired vs NS-P8 at the same budget where it exists (same test queries)
base = {(p['task'], round(p['budget'], 4)): p for p in pts if p['family'] == 'NS-P8'}
for p in pts:
    b = base.get((p['task'], round(p['budget'], 4)))
    if b is not None and p is not b and p['family'] != 'FullKV':
        try: p.update({f'vsNS_{k}': v for k, v in paired(b['_rows'], p['_rows']).items() if k in ('delta_pp', 'harmed', 'rescued', 'disagree')})
        except ValueError as e: p['vsNS_error'] = str(e)
keys = ['task', 'family', 'label', 'budget', 'period', 'acc', 'MB', 'selected', 'vsNS_delta_pp', 'vsNS_harmed', 'vsNS_rescued', 'vsNS_disagree', 'run']
with (OUT / 'quality_points.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(pts)

# NS P4 vs P8 (dev + test)
rows = []
for t in TASKS:
    for b in (5, 10, 25, 50):
        for split, p8, p4 in (('dev', V29 / f'a0/runs/devI_{t}_b{b}', Q / f'ns_p4_dev_{t}_k{b}'),
                              ('test', V29 / f'test_quality/runs/test_I_{t}_b{b}', Q / f'ns_p4_test_{t}_k{b}')):
            a, c = try_load(p8), try_load(p4)
            if a and c:
                pr = paired(a[0], c[0])
                rows.append(dict(task=t, budget=b, split=split, acc_P8=100 * pr['acc_base'], acc_P4=100 * pr['acc'], delta_pp=pr['delta_pp'],
                                 harmed=pr['harmed'], rescued=pr['rescued'], disagree=pr['disagree'], n=pr['n'],
                                 MB_P8=a[1]['mean_physical_prefetch_kv_bytes'] / 1e6, MB_P4=c[1]['mean_physical_prefetch_kv_bytes'] / 1e6))
if rows:
    with (OUT / 'p4_vs_p8.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# frontier: accuracy of each point minus the NS curve (same period family) linearly interpolated at the same MB
fr = []
for p in pts:
    if p['family'] in ('NS-P8', 'NS-P4', 'FullKV'): continue
    fam = 'NS-P4' if p['period'] == 4 else 'NS-P8'
    curve = sorted((q['MB'], q['acc']) for q in pts if q['task'] == p['task'] and q['family'] == fam)
    if len(curve) < 2 or not curve[0][0] <= p['MB'] <= curve[-1][0]: interp = None
    else:
        for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
            if x0 <= p['MB'] <= x1: interp = y0 + (y1 - y0) * (p['MB'] - x0) / max(x1 - x0, 1e-9); break
    fr.append(dict(task=p['task'], family=p['family'], label=p['label'], MB=p['MB'], acc=p['acc'], ns_curve=fam,
                   ns_acc_at_same_MB=interp, above_curve_pp=None if interp is None else p['acc'] - interp))
with (OUT / 'frontier.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(fr[0])); w.writeheader(); w.writerows(fr)
print('points', len(pts), 'frontier rows', len(fr))

# ---------- timing ----------
T = R / 'studies/v30/timing'
if (T / 'blocks.jsonl').exists():
    blocks = {}
    for b in map(json.loads, (T / 'blocks.jsonl').read_text().splitlines()):
        if b['flag'] or not b['all_ok']: continue
        for r in b['runs']:
            recs = [json.loads(l) for l in (T / 'runs' / r['name'] / 'scored_records.jsonl').read_text().splitlines()]
            assert len(recs) == 16 and len({x['uid'] for x in recs}) == 16
            blocks.setdefault(b['cell'], {}).setdefault(r['arm'], {})[b['block']] = (st.mean(x['ttft_ms'] for x in recs),
                                                                                   st.mean(x['physical_prefetch_kv_bytes'] for x in recs))
    out = []
    for cell, arms in blocks.items():
        ref = arms['NS25']
        for arm, bl in arms.items():
            ks = sorted(set(bl) & set(ref)); ratio = [bl[k][0] / ref[k][0] for k in ks]
            out.append(dict(cell=cell, arm=arm, blocks=len(ks), ttft_ms=st.mean(bl[k][0] for k in ks), MB=st.mean(bl[k][1] for k in ks) / 1e6,
                            vs_NS25_pct=100 * (st.mean(ratio) - 1), faster_blocks=sum(x < 1 for x in ratio)))
    with (OUT / 'timing.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print('timing rows', len(out))
