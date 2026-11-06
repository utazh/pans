"""Apply frozen rule rev 1 to A0 identity + A1 candidate dev runs. Read-only on runs;
writes results/v29_a1/{table.csv,selection.json}. Cells with any missing run are 'pending'."""
from pathlib import Path
import csv, hashlib, json, math, sys
R = Path('/home/shensicheng/qyd/pans/src')
rule_p = R / 'configs/v29_selection_rule.json'
freeze = json.loads((R / 'audit/v29_rule_freeze.json').read_text())
assert hashlib.sha256(rule_p.read_bytes()).hexdigest() == freeze['sha256']
rule = json.loads(rule_p.read_text())
A0, A1 = R / 'studies/v29/a0/runs', R / 'studies/v29/a1/runs'
SHARED = {'I': 0, 'G24': 4, 'G20': 8, 'G16': 12}
ORDER = ['I', 'G24', 'G20', 'G16']

def load(run):
    rec = run / 'scored_records.jsonl'; summ = run / 'summary.json'
    if not rec.exists() or not summ.exists():
        return None
    rows = {r['uid']: r for r in map(json.loads, rec.read_text().splitlines())}
    t = list(json.loads(summ.read_text())['tasks'].values())[0]
    b = t.get('mean_physical_prefetch_kv_bytes')
    if b is None or not math.isfinite(b):
        raise SystemExit(f'{run}: bytes field missing/non-finite -> rule says pause and revise')
    return rows, t['samples'], b

def dist(r):
    s = r['label_candidate_mean_logprobs']; ks = sorted(s); m = max(s.values())
    e = [math.exp(s[k] - m) for k in ks]; z = sum(e)
    return [x / z for x in e]

def kl(p, q):
    return sum(a * (math.log(a) - math.log(max(b, 1e-30))) for a, b in zip(p, q) if a > 0)

table, selection = [], {}
for cell in rule['cells']:
    task, b = cell.split('_'); b = int(b[1:])
    runs = {'I': load(A0 / f'devI_{cell}')}
    for c in ('G16', 'G20', 'G24'):
        runs[c] = load(A1 / f'dev_{c}_{cell}')
    if any(v is None for v in runs.values()):
        selection[cell] = dict(status='pending'); continue
    I_rows, nI, I_bytes = runs['I']
    tol = math.ceil(0.01 * nI)
    cands = {}
    for name in ORDER:
        rows, n, byts = runs[name]
        assert set(rows) == set(I_rows) and n == nI, (cell, name)
        cI = sum(r['correct'] for r in I_rows.values()); cC = sum(rows[u]['correct'] for u in I_rows)
        dis = sum(rows[u]['prediction'] != I_rows[u]['prediction'] for u in I_rows)
        harm = sum(I_rows[u]['correct'] and not rows[u]['correct'] for u in I_rows)
        good = sum(rows[u]['correct'] and not I_rows[u]['correct'] for u in I_rows)
        mkl = sum(kl(dist(I_rows[u]), dist(rows[u])) for u in I_rows) / nI
        net = cC - cI
        eligible = name == 'I' or net >= -tol
        cands[name] = dict(correct=cC, n=nI, acc=cC / nI, net=net, delta_pp=100 * net / nI, eligible=eligible,
                           bytes=byts, bytes_vs_I=byts / I_bytes - 1, disagree=dis, harmful=harm, beneficial=good,
                           mean_kl=mkl, shared_consumers=SHARED[name])
        table.append(dict(cell=cell, cand=name, **cands[name]))
    elig = [k for k in ORDER if cands[k]['eligible']]
    bmin = min(cands[k]['bytes'] for k in elig)
    near = [k for k in elig if (cands[k]['bytes'] - bmin) / bmin < 0.005]
    pick = sorted(near, key=lambda k: (-cands[k]['net'], cands[k]['shared_consumers'], ORDER.index(k)))[0]
    selection[cell] = dict(status='selected', selected=pick, tolerance_correct=tol,
                           no_qualified_compression_point=(pick == 'I'), eligible=elig, near_min=near,
                           selected_bytes_vs_I=cands[pick]['bytes_vs_I'], selected_delta_pp=cands[pick]['delta_pp'])

out = R / 'results/v29_a1'; out.mkdir(parents=True, exist_ok=True)
with (out / 'table.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(table[0]) if table else ['cell']); w.writeheader(); w.writerows(table)
(out / 'selection.json').write_text(json.dumps(dict(rule_sha256=freeze['sha256'], cells=selection), indent=1))
print('cell       cand  acc%   dPP    bytes%  dis  harm good  KL      elig')
for r in table:
    print(f"{r['cell']:9s} {r['cand']:4s} {100*r['acc']:6.2f} {r['delta_pp']:+6.2f} {100*r['bytes_vs_I']:+7.1f} "
          f"{r['disagree']:4d} {r['harmful']:4d} {r['beneficial']:4d} {r['mean_kl']:.4f}  {r['eligible']}")
print()
for cell, s in selection.items():
    print(cell, s if s['status'] == 'pending' else
          f"-> {s['selected']}  dPP {s['selected_delta_pp']:+.2f}  bytes {100*s['selected_bytes_vs_I']:+.1f}%  eligible {s['eligible']}")
