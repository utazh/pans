"""v29 P0-b: test-set quality of our read-side sharing (G24/G20/G16 and the dev-selected config) vs the
prefix-adapted cross-layer baselines from v28 (KVSharer-Prefix G24/G20/G16, xKV-CLF/P8 4 variants).
Every row is bound to its run directory. External baselines are quality bridges (no speed meaning).
Writes results/v29_crosslayer/{table.csv,summary.json}."""
from pathlib import Path
import csv, json, statistics as st
R = Path('/home/panzihang/src/prism_layerkv')
final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
V28, TQ, P0 = R / 'studies/v28_frozen/runs', R / 'studies/v29/test_quality/runs', R / 'studies/v29/p0_test_g/runs'
def load(p):
    rows = {r['uid']: r for r in map(json.loads, (p / 'scored_records.jsonl').read_text().splitlines())}
    return rows
def ours(cell, g):
    sel = final[cell]['config']
    if sel == g: return TQ / f'test_{g}_{cell}'
    return V28 / f'{cell}_quality_R0' if g == 'G16' else P0 / f'test_{g}_{cell}'
METHODS = {'ours_G24': lambda c: ours(c, 'G24'), 'ours_G20': lambda c: ours(c, 'G20'), 'ours_G16': lambda c: ours(c, 'G16'),
           'ours_selected': lambda c: TQ / f"test_{final[c]['config']}_{c}",
           'kvsharer_G24': lambda c: V28 / f'{c}_external_quality_kvsharer_g24', 'kvsharer_G20': lambda c: V28 / f'{c}_external_quality_kvsharer_g20',
           'kvsharer_G16': lambda c: V28 / f'{c}_external_quality_kvsharer_g16',
           'xkv_256_384_fp32': lambda c: V28 / f'{c}_external_quality_clf_256_384_fp32', 'xkv_256_384_bf16': lambda c: V28 / f'{c}_external_quality_clf_256_384_bf16',
           'xkv_384_576_fp32': lambda c: V28 / f'{c}_external_quality_clf_384_576_fp32', 'xkv_384_576_bf16': lambda c: V28 / f'{c}_external_quality_clf_384_576_bf16'}
table = []
for cell in final:
    I = load(TQ / f'test_I_{cell}'); n = len(I); cI = sum(r['correct'] for r in I.values())
    for m, path in METHODS.items():
        p = path(cell)
        if m == 'ours_selected' and final[cell]['config'] == 'I': rows = I
        else: rows = load(p)
        assert set(rows) == set(I), (cell, m)
        c = sum(rows[u]['correct'] for u in I)
        table.append(dict(cell=cell, method=m, run=str(p.relative_to(R)), n=n, acc=100 * c / n, I_acc=100 * cI / n, delta_pp=100 * (c - cI) / n,
                          disagree_pct=100 * sum(rows[u]['prediction'] != I[u]['prediction'] for u in I) / n))
out = R / 'results/v29_crosslayer'; out.mkdir(parents=True, exist_ok=True)
with (out / 'table.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
by = lambda m: {r['cell']: r for r in table if r['method'] == m}
summ = {}
for m in METHODS:
    d = by(m); nt = [v for c, v in d.items() if not c.startswith('trec')]
    summ[m] = dict(mean_delta=st.mean(v['delta_pp'] for v in d.values()), mean_delta_non_trec=st.mean(v['delta_pp'] for v in nt),
                   worst=min(v['delta_pp'] for v in d.values()), within_1pp=sum(v['delta_pp'] >= -1 for v in d.values()),
                   mean_disagree=st.mean(v['disagree_pct'] for v in d.values()), mean_acc=st.mean(v['acc'] for v in d.values()))
wins = {}
for a, b in [('ours_G24', 'kvsharer_G24'), ('ours_G20', 'kvsharer_G20'), ('ours_G16', 'kvsharer_G16'),
             ('ours_selected', 'xkv_256_384_bf16'), ('ours_selected', 'xkv_384_576_bf16'), ('ours_G16', 'xkv_256_384_bf16'), ('ours_G24', 'xkv_256_384_bf16')]:
    A, B = by(a), by(b)
    wins[f'{a}>{b}'] = dict(higher=sum(A[c]['acc'] > B[c]['acc'] for c in A), tie=sum(A[c]['acc'] == B[c]['acc'] for c in A),
                            lower=sum(A[c]['acc'] < B[c]['acc'] for c in A), mean_diff=st.mean(A[c]['acc'] - B[c]['acc'] for c in A))
(out / 'summary.json').write_text(json.dumps(dict(methods=summ, pairwise=wins), indent=1))
print(f"{'method':18s} meanAcc  meanD  nonTREC  worst  <=1pp  disagree%")
for m, s in summ.items():
    print(f"{m:18s} {s['mean_acc']:6.2f} {s['mean_delta']:+6.2f} {s['mean_delta_non_trec']:+7.2f} {s['worst']:+7.2f} {s['within_1pp']:4d}/16 {s['mean_disagree']:7.1f}")
for k, v in wins.items(): print(f"{k:34s} higher {v['higher']:2d} tie {v['tie']} lower {v['lower']:2d}  mean diff {v['mean_diff']:+.2f} pp")
print('\nper-cell accuracy:'); ms = list(METHODS)
print('cell      I     ' + ' '.join(m.replace('ours_', 'o').replace('kvsharer_', 'kv').replace('xkv_', 'x')[:9].rjust(9) for m in ms))
for cell in final:
    print(f"{cell:9s} {by(ms[0])[cell]['I_acc']:5.1f} " + ' '.join(f"{by(m)[cell]['acc']:9.1f}" for m in ms))
