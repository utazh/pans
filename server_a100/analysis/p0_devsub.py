"""P0: how small can the dev set be?  Re-apply the frozen v29 selection rule
(net >= -ceil(0.01 m), min bytes, 0.5% tie, then -net, consumers, arm order)
to random m-query subsets of the saved 500-query dev runs.  No GPU.
Output: agreement with the full-500 choice and the test outcome of the
subset-selected arm (test runs exist for every arm)."""
import json, math, random, sys
from pathlib import Path

BASE = Path('/home/shensicheng/qyd/pans')
ROOTS = {'7B': BASE / 'src/studies/v29m', '14B': BASE / 'src_q14/studies/v29m_s800', '32B': BASE / 'src_q32/studies/v29m_s800'}
ARMS = ('I', 'S1', 'S2', 'S3')
ORDER = {'I': 0, 'S1': 1, 'S2': 2, 'S3': 3}
SIZES = (50, 100, 150, 200, 300, 500)
DRAWS = 2000


def load(d):
    recs = [json.loads(l) for l in open(d / 'scored_records.jsonl')]
    summ = json.loads((d / 'summary.json').read_text())
    return ({r['uid']: bool(r['correct']) for r in recs},
            list(summ['tasks'].values())[0]['mean_physical_prefetch_kv_bytes'])


def select(sub, dev, nbytes, cons):
    m = len(sub); tol = math.ceil(0.01 * m)
    base = sum(dev['I'][u] for u in sub)
    rows = {}
    for a in ARMS:
        net = sum(dev[a][u] for u in sub) - base
        if a == 'I' or net >= -tol:
            rows[a] = (net, nbytes[a], cons[a])
    bmin = min(v[1] for v in rows.values())
    near = [a for a, v in rows.items() if (v[1] - bmin) / bmin < 0.005]
    return sorted(near, key=lambda a: (-rows[a][0], rows[a][2], ORDER[a]))[0]


out = {}
rng = random.Random(20261007)
for model, W in ROOTS.items():
    for sf in sorted((W / 'selection').glob('*.json')):
        cell = sf.stem; full = json.loads(sf.read_text())
        dev, nb = {}, {}
        for a in ARMS:
            dev[a], nb[a] = load(W / 'dev' / f'{cell}_dev_{a}')
        cons = {a: full['candidates'][a]['consumers'] for a in ARMS}
        test = {}
        for a in ARMS:
            d = W / 'test' / f'{cell}_test_{a}'
            if (d / 'scored_records.jsonl').exists():
                t, b = load(d); test[a] = dict(acc=100 * sum(t.values()) / len(t), bytes=b)
        uids = sorted(dev['I'])
        assert all(sorted(dev[a]) == uids for a in ARMS), cell
        full_sel = select(uids, dev, nb, cons)
        assert full_sel == full['selected'], (cell, full_sel, full['selected'])
        cell_out = dict(full=full_sel, n=len(uids), test=test, sizes={})
        for m in SIZES:
            cnt = {a: 0 for a in ARMS}
            for _ in range(DRAWS if m < len(uids) else 1):
                cnt[select(rng.sample(uids, m), dev, nb, cons)] += 1
            tot = sum(cnt.values())
            frac = {a: c / tot for a, c in cnt.items()}
            row = dict(agree=frac[full_sel], dist=frac)
            if len(test) == 4:
                I = test['I']
                row['exp_test_dpp'] = sum(frac[a] * (test[a]['acc'] - I['acc']) for a in ARMS)
                row['exp_bytes_vs_I'] = sum(frac[a] * (test[a]['bytes'] / I['bytes'] - 1) for a in ARMS)
                row['p_test_loss_gt1pp'] = sum(frac[a] for a in ARMS if test[a]['acc'] - I['acc'] < -1.0)
            cell_out['sizes'][m] = row
        out[f'{model}/{cell}'] = cell_out
        print(model, cell, 'full=', full_sel,
              {a: round(test[a]['acc'] - test['I']['acc'], 2) for a in test} if len(test) == 4 else 'no-test',
              flush=True)
        for m, r in cell_out['sizes'].items():
            print(f'   m={m:3d} agree={r["agree"]:.3f} dist=' + ' '.join(f'{a}:{r["dist"][a]:.2f}' for a in ARMS)
                  + (f' E[dpp]={r["exp_test_dpp"]:+.2f} E[bytes]={100*r["exp_bytes_vs_I"]:+.1f}% P(loss>1pp)={r["p_test_loss_gt1pp"]:.3f}'
                     if 'exp_test_dpp' in r else ''), flush=True)
Path(sys.argv[1] if len(sys.argv) > 1 else BASE / 'analysis/p0_devsub.json').write_text(json.dumps(out, indent=1))
