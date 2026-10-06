"""v29 stage C report: baselines (ContiguousKV, IMPRESS) vs our system on test accuracy and TTFT.
Stage C blocks contain I/CKV/IMP; stage B blocks contain I/G16/selected. Our selected vs a baseline is
(sel/I)_B x (I/base)_C using the shared I arm as bridge; I drift between sessions is reported.
Writes results/v29_stage_c/{table.csv,summary.json}."""
from pathlib import Path
import csv, json, math, random, statistics as st
R = Path('/home/shensicheng/qyd/pans/src')
final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
tq = {r['cell']: r for r in csv.DictReader((R / 'results/v29_test_quality/table.csv').open())}

def block_means(root):
    out = {}
    for b in map(json.loads, (root / 'blocks.jsonl').read_text().splitlines()):
        if b['flag'] or not b['all_ok']: continue
        for r in b['runs']:
            t = [json.loads(l)['ttft_ms'] for l in (root / 'runs' / r['name'] / 'scored_records.jsonl').read_text().splitlines()]
            assert len(t) == 16
            out.setdefault((b['cell'], b['cond']), {}).setdefault(r['arm'], {})[b['block']] = st.mean(t)
    return out
B, C = block_means(R / 'studies/v29/timing'), block_means(R / 'studies/v29/timing_c')

cold_bad = 0
for root in (R / 'studies/v29/timing_c',):
    for p in (root / 'runs').glob('*.cold.jsonl'):
        cold_bad += sum(json.loads(l).get('status') != 'completed' for l in p.read_text().splitlines())

def acc(m, cell):
    t, b = cell.split('_')
    s = json.loads((R / f'studies/v29/c_quality/runs/{m}_{t}_{b}/summary.json').read_text())
    return 100 * list(s['tasks'].values())[0]['accuracy']

def boot(xs, n=4000):
    rng = random.Random(0); m = sorted(st.mean(rng.choice(xs) for _ in xs) for _ in range(n)); return m[int(.025 * n)], m[int(.975 * n)]

rows = []
for (cell, cond), c in sorted(C.items()):
    b = B.get((cell, cond)); sel = final[cell]['config']
    blocks = sorted(c['I'])
    ratio = lambda arm: [c[arm][k] / c['I'][k] for k in blocks]  # base / I, per block (stage C)
    ckv, imp = ratio('CKV'), ratio('IMP')
    sel_over_I = 1.0 if sel == 'I' else st.mean(b[sel][k] / b['I'][k] for k in sorted(b['I']))
    row = dict(cell=cell, cond=cond, selected=sel,
               I_ms_C=st.mean(c['I'].values()), I_ms_B=st.mean(b['I'].values()),
               I_drift_pct=100 * (st.mean(c['I'].values()) / st.mean(b['I'].values()) - 1),
               CKV_ms=st.mean(c['CKV'].values()), IMP_ms=st.mean(c['IMP'].values()),
               sel_ms_est=sel_over_I * st.mean(c['I'].values()),
               I_vs_CKV_pct=100 * (1 / st.mean(ckv) - 1), I_vs_CKV_ci=[100 * (1 / x - 1) for x in boot(ckv)[::-1]],
               I_vs_IMP_pct=100 * (1 / st.mean(imp) - 1),
               sel_vs_CKV_pct=100 * (sel_over_I / st.mean(ckv) - 1), sel_vs_IMP_pct=100 * (sel_over_I / st.mean(imp) - 1),
               I_faster_than_CKV_blocks=sum(x > 1 for x in ckv), blocks=len(blocks),
               acc_CKV=acc('contigkv', cell), acc_IMP=acc('impress', cell),
               acc_I=float(tq[cell]['I_acc']), acc_sel=float(tq[cell]['sel_acc']))
    rows.append(row)
out = R / 'results/v29_stage_c'; out.mkdir(parents=True, exist_ok=True)
with (out / 'table.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
cold = [r for r in rows if r['cond'] == 'preserve']
mean = lambda k, xs: st.mean(r[k] for r in xs)
summ = dict(cold_not_completed=cold_bad,
            cold=dict(I_vs_CKV=mean('I_vs_CKV_pct', cold), I_vs_IMP=mean('I_vs_IMP_pct', cold),
                      sel_vs_CKV=mean('sel_vs_CKV_pct', cold), sel_vs_IMP=mean('sel_vs_IMP_pct', cold),
                      I_drift_mean=mean('I_drift_pct', cold), I_drift_max_abs=max(abs(r['I_drift_pct']) for r in cold)),
            warm=dict(sel_vs_CKV=mean('sel_vs_CKV_pct', [r for r in rows if r['cond'] == 'warm']),
                      sel_vs_IMP=mean('sel_vs_IMP_pct', [r for r in rows if r['cond'] == 'warm'])),
            accuracy=dict(mean_CKV=mean('acc_CKV', cold), mean_IMP=mean('acc_IMP', cold), mean_I=mean('acc_I', cold),
                          mean_sel=mean('acc_sel', cold)))
(out / 'summary.json').write_text(json.dumps(summ, indent=1))
print(f"{'cell':9s} {'cond':8s} sel  | acc CKV  IMP    I    sel  | ms CKV    IMP     I    sel~  | I-drift  sel/CKV  sel/IMP")
for r in rows:
    print(f"{r['cell']:9s} {r['cond']:8s} {r['selected']:4s} | {r['acc_CKV']:5.1f} {r['acc_IMP']:5.1f} {r['acc_I']:5.1f} {r['acc_sel']:5.1f} | "
          f"{r['CKV_ms']:6.0f} {r['IMP_ms']:7.0f} {r['I_ms_C']:6.0f} {r['sel_ms_est']:6.0f} | {r['I_drift_pct']:+6.1f}% {r['sel_vs_CKV_pct']:+7.1f}% {r['sel_vs_IMP_pct']:+7.1f}%")
print(json.dumps(summ, indent=1))
