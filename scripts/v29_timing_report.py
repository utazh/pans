"""v29 stage B analysis: block-paired TTFT per cell/condition/arm + test-set accuracy.
Writes results/v29_timing/{table.csv,summary.json}. Also audits cold certification."""
from pathlib import Path
import csv, json, math, random, statistics as st
R = Path('/home/shensicheng/qyd/pans/src'); T = R / 'studies/v29/timing'
blocks = [json.loads(l) for l in (T / 'blocks.jsonl').read_text().splitlines()]
final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
tq = {r['cell']: r for r in csv.DictReader((R / 'results/v29_test_quality/table.csv').open())}
v28 = lambda cell, a: 100 * list(json.loads((R / f'studies/v28_frozen/runs/{cell}_quality_{a}/summary.json').read_text())['tasks'].values())[0]['accuracy']

cold_bad, cold_n = [], 0
data = {}  # (cell, cond) -> arm -> [block means]
for b in blocks:
    if b['flag'] or not b['all_ok']: continue
    for r in b['runs']:
        run = T / 'runs' / r['name']
        ttft = [json.loads(l)['ttft_ms'] for l in (run / 'scored_records.jsonl').read_text().splitlines()]
        assert len(ttft) == 16, r['name']
        data.setdefault((b['cell'], b['cond']), {}).setdefault(r['arm'], {})[b['block']] = st.mean(ttft)
        if b['cond'] == 'preserve':
            for c in map(json.loads, Path(f'{run}.cold.jsonl').read_text().splitlines()):
                cold_n += 1
                if c.get('status') != 'completed': cold_bad.append((r['name'], c.get('status')))

def boot_ci(diffs, n=4000, seed=0):
    rng = random.Random(seed); m = []
    for _ in range(n):
        s = [rng.choice(diffs) for _ in diffs]; m.append(st.mean(s))
    m.sort(); return m[int(0.025 * n)], m[int(0.975 * n)]

rows = []
for (cell, cond), arms in data.items():
    I = arms['I']; blocks_ = sorted(I)
    sel = final[cell]['config']
    for arm, v in arms.items():
        if arm == 'I': continue
        rel = [v[k] / I[k] - 1 for k in blocks_]
        lo, hi = boot_ci(rel)
        acc = float(tq[cell]['sel_acc']) if arm == sel else (v28(cell, 'R0') if arm == 'G16' else None)
        rows.append(dict(cell=cell, cond=cond, arm=arm, selected=(arm == sel), I_ms=st.mean(I.values()), arm_ms=st.mean(v.values()),
                         rel_pct=100 * st.mean(rel), ci_lo=100 * lo, ci_hi=100 * hi, blocks_faster=sum(x < 0 for x in rel), blocks=len(rel),
                         I_acc=float(tq[cell]['I_acc']), arm_acc=acc, dacc_pp=None if acc is None else acc - float(tq[cell]['I_acc']),
                         bytes_vs_I=float(tq[cell]['bytes_vs_I']) if arm == sel else None))
    if sel == 'I':
        rows.append(dict(cell=cell, cond=cond, arm='I(sel)', selected=True, I_ms=st.mean(I.values()), arm_ms=st.mean(I.values()),
                         rel_pct=0.0, ci_lo=0.0, ci_hi=0.0, blocks_faster=0, blocks=len(I), I_acc=float(tq[cell]['I_acc']),
                         arm_acc=float(tq[cell]['I_acc']), dacc_pp=0.0, bytes_vs_I=0.0))
out = R / 'results/v29_timing'; out.mkdir(parents=True, exist_ok=True)
with (out / 'table.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def agg(cond, pred):
    xs = [r for r in rows if r['cond'] == cond and pred(r)]
    return dict(cells=len(xs), mean_rel_pct=st.mean(r['rel_pct'] for r in xs), min_rel=min(r['rel_pct'] for r in xs),
                max_rel=max(r['rel_pct'] for r in xs), ci_excludes_0_faster=sum(r['ci_hi'] < 0 for r in xs),
                ci_excludes_0_slower=sum(r['ci_lo'] > 0 for r in xs))
summ = dict(cold_requests=cold_n, cold_not_completed=cold_bad[:10], cold_not_completed_n=len(cold_bad),
            selected_cold=agg('preserve', lambda r: r['selected']), selected_warm=agg('warm', lambda r: r['selected']),
            g16_cold=agg('preserve', lambda r: r['arm'] == 'G16'), g16_warm=agg('warm', lambda r: r['arm'] == 'G16'))
(out / 'summary.json').write_text(json.dumps(summ, indent=1))
print(f"{'cell':9s} {'cond':8s} {'arm':6s} {'I ms':>8s} {'arm ms':>8s} {'rel%':>7s} {'95%CI':>16s} fast/6  accI  accArm  dAcc  bytes%")
for r in sorted(rows, key=lambda r: (r['cond'] != 'preserve', r['cell'], not r['selected'], r['arm'])):
    da = '' if r['dacc_pp'] is None else f"{r['dacc_pp']:+5.2f}"
    aa = '' if r['arm_acc'] is None else f"{r['arm_acc']:6.2f}"
    by = '' if r['bytes_vs_I'] is None else f"{r['bytes_vs_I']:+6.1f}"
    print(f"{r['cell']:9s} {r['cond']:8s} {r['arm']+('*' if r['selected'] else ''):6s} {r['I_ms']:8.1f} {r['arm_ms']:8.1f} {r['rel_pct']:+7.1f} "
          f"[{r['ci_lo']:+6.1f},{r['ci_hi']:+6.1f}] {r['blocks_faster']}/{r['blocks']}  {r['I_acc']:5.2f} {aa:>6s} {da:>6s} {by:>6s}")
print(json.dumps(summ, indent=1))
