"""Test-set quality summary for the frozen v29 selection (I vs selected), with v28 G16/R0 and Full-KV
references. Writes results/v29_test_quality/{table.csv,summary.json}."""
from pathlib import Path
import csv, json, math
R = Path('/home/shensicheng/qyd/pans/src')
final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
T = R / 'studies/v29/test_quality/runs'

def load(p):
    rows = {r['uid']: r for r in map(json.loads, (p / 'scored_records.jsonl').read_text().splitlines())}
    return rows, list(json.loads((p / 'summary.json').read_text())['tasks'].values())[0]

def dist(r):
    s = r['label_candidate_mean_logprobs']; ks = sorted(s); m = max(s.values())
    e = [math.exp(s[k] - m) for k in ks]; z = sum(e); return [x / z for x in e]

def kl(p, q): return sum(a * (math.log(a) - math.log(max(b, 1e-30))) for a, b in zip(p, q) if a > 0)

out, table = R / 'results/v29_test_quality', []
out.mkdir(parents=True, exist_ok=True)
full = {t: load(R / f'studies/v29/a0/runs/fullkv_{t}')[1]['accuracy'] for t in ('sst2', 'subj', 'trec', 'rte')}
for cell, f in final.items():
    task = cell.split('_')[0]
    I, It = load(T / f'test_I_{cell}'); n = len(I); cI = sum(r['correct'] for r in I.values())
    v28 = {a: load(R / f'studies/v28_frozen/runs/{cell}_quality_{a}') for a in ('I', 'R0')}
    row = dict(cell=cell, config=f['config'], n=n, fullkv_acc=100 * full[task], I_acc=100 * cI / n,
               v28_I_acc=100 * v28['I'][1]['accuracy'], v28_G16_acc=100 * v28['R0'][1]['accuracy'])
    if f['config'] == 'I':
        row.update(sel_acc=row['I_acc'], delta_pp=0.0, net=0, disagree=0, harmful=0, beneficial=0, mean_kl=0.0, bytes_vs_I=0.0)
    else:
        S, St = load(T / f"test_{f['config']}_{cell}"); assert set(S) == set(I)
        cS = sum(S[u]['correct'] for u in I)
        row.update(sel_acc=100 * cS / n, delta_pp=100 * (cS - cI) / n, net=cS - cI,
                   disagree=sum(S[u]['prediction'] != I[u]['prediction'] for u in I),
                   harmful=sum(I[u]['correct'] and not S[u]['correct'] for u in I),
                   beneficial=sum(S[u]['correct'] and not I[u]['correct'] for u in I),
                   mean_kl=sum(kl(dist(I[u]), dist(S[u])) for u in I) / n,
                   bytes_vs_I=100 * (St['mean_physical_prefetch_kv_bytes'] / It['mean_physical_prefetch_kv_bytes'] - 1))
    row['v28_G16_delta_pp'] = row['v28_G16_acc'] - row['v28_I_acc']
    table.append(row)
with (out / 'table.csv').open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
mean = lambda k: sum(r[k] for r in table) / len(table)
summ = dict(cells=len(table), mean_delta_pp=mean('delta_pp'), mean_bytes_vs_I=mean('bytes_vs_I'),
            v28_G16_mean_delta_pp=mean('v28_G16_delta_pp'), worst_delta_pp=min(r['delta_pp'] for r in table),
            cells_within_1pp=sum(r['delta_pp'] >= -1.0 for r in table),
            non_trec_mean_delta_pp=sum(r['delta_pp'] for r in table if not r['cell'].startswith('trec')) / 12)
(out / 'summary.json').write_text(json.dumps(summ, indent=1))
print('cell      cfg  Full   I      sel    dPP    net  dis  KL     bytes%  | v28 I  v28G16 dG16')
for r in table:
    print(f"{r['cell']:9s} {r['config']:4s} {r['fullkv_acc']:5.1f} {r['I_acc']:6.2f} {r['sel_acc']:6.2f} {r['delta_pp']:+6.2f} "
          f"{r['net']:+4d} {r['disagree']:4d} {r['mean_kl']:.3f} {r['bytes_vs_I']:+7.1f} | {r['v28_I_acc']:6.2f} {r['v28_G16_acc']:6.2f} {r['v28_G16_delta_pp']:+5.2f}")
print(json.dumps(summ, indent=1))
