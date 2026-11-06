"""v30 QuALITY report (read-only). 12 articles (sets f0/f1/f2 x 4 slots), all real questions. Pools questions over
articles; 95% CIs by article-level bootstrap (articles resampled, not questions). Writes results/v30/quality_qa.csv and
quality_qa_summary.json. Arms: v29 FULLKV / I (NS P8 25) / syn_G24 / syn_G20 / syn_G16 and v30 NSP4_25, NS_<b>, CAKE_<b>, LAVA_<b>."""
from pathlib import Path
import csv, json, random, runpy
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R = C['R']; SETS = ['f0', 'f1', 'f2']; SLOTS = C['TASKS']
V29 = lambda s: R / f'studies/v29/quality/{s}/eval'; V30 = R / 'studies/v30/quality_qa/runs'
EQ = {'21p4286': 'G24', '17p8571': 'G20', '14p2857': 'G16'}
ARMS = {'FULLKV': ('v29', 'FULLKV'), 'NS25': ('v29', 'I'), 'G24': ('v29', 'syn_G24'), 'G20': ('v29', 'syn_G20'), 'G16': ('v29', 'syn_G16'),
        'NSP4_25': ('v30', 'NSP4_25')}
for t in EQ: ARMS[f'NS_{t}'] = ('v30', f'NS_{t}')
for m in ('CAKE', 'LAVA'):
    for t in ('25', *EQ): ARMS[f'{m}_{t}'] = ('v30', f'{m}_{t}')
for g in ('G24', 'G21', 'G20', 'G16'): ARMS[f'RAPSP4_{g}'] = ('p4', f'RAPSP4_{g}')
for t in EQ: ARMS[f'NSP4_{t}'] = ('p4', f'NSP4_{t}')

def run_dir(src, name, s, slot):
    if src == 'p4': return R / f'studies/v30/qa_p4/{s}_{slot}/eval/{name}'
    return V29(s) / f'{name}_{slot}' if src == 'v29' else V30 / f'{s}_{slot}_{name}'

data = {}   # arm -> article -> (rows, MB)
for arm, (src, name) in ARMS.items():
    for s in SETS:
        for slot in SLOTS:
            d = run_dir(src, name, s, slot)
            if not (d / 'summary.json').exists(): continue
            rows = {json.loads(l)['uid']: json.loads(l) for l in (d / 'scored_records.jsonl').read_text().splitlines()}
            t = list(json.loads((d / 'summary.json').read_text())['tasks'].values())[0]
            data.setdefault(arm, {})[f'{s}_{slot}'] = (rows, (t.get('mean_physical_prefetch_kv_bytes') or float('nan')) / 1e6)
arts = sorted(data['NS25'])
def acc(arm, A):
    rows = [r for a in A if a in data[arm] for r in data[arm][a][0].values()]; return 100 * sum(r['correct'] for r in rows) / len(rows)
def delta_ci(arm, base, n=4000):
    A = [a for a in arts if a in data.get(arm, {}) and a in data.get(base, {})]
    for a in A:
        if set(data[arm][a][0]) != set(data[base][a][0]): raise ValueError(f'{arm}/{base} {a}: question sets differ')
    rng = random.Random(0); xs = []
    for _ in range(n):
        S = [rng.choice(A) for _ in A]; xs.append(acc(arm, S) - acc(base, S))
    xs.sort(); return acc(arm, A) - acc(base, A), xs[int(.025 * n)], xs[int(.975 * n)], len(A)
rows = []
for arm in ARMS:
    if arm not in data: continue                      # arms on a subset of articles (P4 fallback sets) report their own count
    A = [a for a in arts if a in data[arm]]
    mb = sum(data[arm][a][1] for a in A) / len(A)
    r = dict(arm=arm, articles=len(A), questions=sum(len(data[arm][a][0]) for a in A), acc=acc(arm, A), MB=mb)
    for base in ('NS25', 'NSP4_25'):
        if base in data and arm != base and set(A) <= set(data[base]):
            d, lo, hi, _ = delta_ci(arm, base); r[f'vs_{base}'] = d; r[f'vs_{base}_ci'] = [lo, hi]
    rows.append(r)
# byte-matched: each RAPS / CAKE / LAVa point vs NS at the same budget
matched = []
for t, g in EQ.items():
    for arm in (g, f'CAKE_{t}', f'LAVA_{t}'):
        if arm in data and f'NS_{t}' in data:
            d, lo, hi, n = delta_ci(arm, f'NS_{t}'); matched.append(dict(budget_tag=t, arm=arm, vs=f'NS_{t}', delta=d, ci=[lo, hi], articles=n))
    if f'RAPSP4_{g}' in data and f'NSP4_{t}' in data:   # P4 byte-matched: RAPS-P4 G<k> vs NS-P4 at the same bytes
        d, lo, hi, n = delta_ci(f'RAPSP4_{g}', f'NSP4_{t}'); matched.append(dict(budget_tag=t, arm=f'RAPSP4_{g}', vs=f'NSP4_{t}', delta=d, ci=[lo, hi], articles=n))
out = R / 'results/v30'; out.mkdir(parents=True, exist_ok=True)
with (out / 'quality_qa.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99))
    w.writeheader(); w.writerows(rows)
(out / 'quality_qa_summary.json').write_text(json.dumps(dict(articles=arts, arms=rows, byte_matched=matched), indent=1))
for r in rows:
    print(f"{r['arm']:12s} acc {r['acc']:6.2f}  MB {r['MB']:7.2f}  vsNS25 {r.get('vs_NS25', 0):+6.2f} {[round(x, 1) for x in r.get('vs_NS25_ci', [])]}"
          f"  vsNSP4 {r.get('vs_NSP4_25', 0):+6.2f}")
print('byte-matched:'); [print(m) for m in matched]
