"""P0b: exact (selection-preserving) ways to cut dev cost.
 (a) cheapest-first: evaluate S3 -> S2 -> S1, stop at the first eligible arm.
     Same choice as the full rule whenever adjacent arms differ by >0.5% bytes.
 (b) (a) + deterministic early termination inside an arm: with k of n queries
     done, eligibility is decided once net_k - (n-k) >= -tol (surely eligible)
     or net_k + (n-k) < -tol (surely not).  Still exactly the same choice.
Cost unit = query-evaluations; full rule = n (I) + 3n (S1..S3)."""
import json, math, random
from pathlib import Path
exec(Path(__file__).with_name('p0_devsub.py').read_text().split('\nout = {}')[0])  # reuse ROOTS/load/select

rng = random.Random(7)
res = {}
for model, W in ROOTS.items():
    for sf in sorted((W / 'selection').glob('*.json')):
        cell = sf.stem; full = json.loads(sf.read_text())
        dev = {a: load(W / 'dev' / f'{cell}_dev_{a}')[0] for a in ARMS}
        nb = {a: full['candidates'][a]['bytes'] for a in ARMS}
        uids = sorted(dev['I']); n = len(uids); tol = math.ceil(0.01 * n)
        gaps_ok = all(abs(nb[x] - nb[y]) / min(nb[x], nb[y]) >= 0.005 for x, y in (('S3', 'S2'), ('S2', 'S1'), ('S1', 'I')))
        # (a)
        arms_a = 0; sel_a = 'I'
        for a in ('S3', 'S2', 'S1'):
            arms_a += 1
            if full['candidates'][a]['eligible']:
                sel_a = a; break
        cost_full = 4 * n; cost_a = n + arms_a * n
        # (b) averaged over random query orders
        costs_b = []
        for _ in range(200):
            order = uids[:]; rng.shuffle(order)
            used_I = 0; total = 0; sel_b = 'I'
            for a in ('S3', 'S2', 'S1'):
                net = 0; dec = None
                for k, u in enumerate(order, 1):
                    net += int(dev[a][u]) - int(dev['I'][u])
                    r = n - k
                    if net - r >= -tol: dec = True; break
                    if net + r < -tol: dec = False; break
                total += k; used_I = max(used_I, k)
                if dec is None: dec = net >= -tol
                if dec: sel_b = a; break
            assert sel_b == sel_a
            costs_b.append(total + used_I)
        cb = sum(costs_b) / len(costs_b)
        res[f'{model}/{cell}'] = dict(selected=full['selected'], cheapest_first_matches=sel_a == full['selected'], byte_gaps_ok=gaps_ok,
                                      arms_evaluated=arms_a, cost_full=cost_full, cost_a=cost_a, cost_b=round(cb))
        print(f'{model:3s} {cell:9s} sel={full["selected"]:2s} match={sel_a == full["selected"]} gaps_ok={gaps_ok} '
              f'arms={arms_a} cost: full {cost_full} | cheapest-first {cost_a} ({cost_a / cost_full:.0%}) | +early-stop {cb:.0f} ({cb / cost_full:.0%})')
Path('/home/shensicheng/qyd/pans/analysis/p0_seq.json').write_text(json.dumps(res, indent=1))
