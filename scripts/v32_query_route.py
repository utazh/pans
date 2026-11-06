"""v32 (3): offline feasibility of per-request sharing decisions (no new GPU runs).
Pairs, per task, the dev500 runs of NS-P4 25 % and of the dev-selected RAPS-P4 configuration (results/v30/p4_selection.json).
Target: the shared run disagrees with NS on the prediction (and the subset where it turns a correct answer wrong).
Candidate signals = per-request selector statistics recorded by the engine in the RAPS run itself (computed before or while
restoring, never from the answer): promixed_mean_uncertainty / boundary_margin / gqa_agreement, selector_mean_jaccard,
impress_period_prediction_mean_jaccard, query length. Reports single-feature AUCs and the routing upper bound (an oracle
that sends exactly the disagreeing requests to native KV) with its byte cost. Usage: v32_query_route.py"""
import json, math, runpy
from pathlib import Path
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R = C['R']; Q = R / 'studies/v30/quality/runs'
sel = json.loads((R / 'results/v30/p4_selection.json').read_text())['cells']
FEATS = ['promixed_mean_uncertainty', 'promixed_mean_boundary_margin', 'promixed_mean_gqa_agreement', 'selector_mean_jaccard',
         'impress_period_prediction_mean_jaccard', 'impress_next_prediction_mean_jaccard']
def auc(xs, ys):
    pos = [x for x, y in zip(xs, ys) if y]; neg = [x for x, y in zip(xs, ys) if not y]
    if not pos or not neg: return None
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg); return s / (len(pos) * len(neg))
out = {}
for t in C['TASKS']:
    g = sel[f'{t}_b25']['selected']
    if g == 'I': continue
    ns, nst = C['load_run'](Q / f'ns_p4_dev_{t}_k25'); sh, sht = C['load_run'](Q / f'p4_dev_{g}_{t}_k25'); C['paired'](ns, sh)
    U = sorted(ns); dis = [ns[u]['prediction'] != sh[u]['prediction'] for u in U]
    harm = [ns[u]['correct'] and not sh[u]['correct'] for u in U]; resc = [sh[u]['correct'] and not ns[u]['correct'] for u in U]
    res = dict(selected=g, n=len(U), disagree=sum(dis), harmed=sum(harm), rescued=sum(resc), acc_NS=100 * sum(ns[u]['correct'] for u in U) / len(U),
               acc_shared=100 * sum(sh[u]['correct'] for u in U) / len(U), feature_auc={})
    for f in FEATS + ['query_len']:
        xs = [len(sh[u].get('query_text', '')) if f == 'query_len' else sh[u].get(f) for u in U]
        if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in xs): continue
        a = auc(xs, dis); b = auc(xs, harm)
        res['feature_auc'][f] = dict(disagree=a, harmed=b)   # 0.5 = no signal; far from 0.5 either way = usable
    # oracle routing upper bound: requests that disagree go to NS (native) -> accuracy and mean bytes
    bn, bs = nst['mean_physical_prefetch_kv_bytes'], sht['mean_physical_prefetch_kv_bytes']; fr = sum(dis) / len(U)
    res['oracle_route'] = dict(fraction_native=fr, acc=100 * sum((ns[u] if d else sh[u])['correct'] for u, d in zip(U, dis)) / len(U),
                               bytes_vs_NS=(fr * bn + (1 - fr) * bs) / bn - 1, shared_bytes_vs_NS=bs / bn - 1)
    out[t] = res
(R / 'results/v32').mkdir(parents=True, exist_ok=True); (R / 'results/v32/query_route.json').write_text(json.dumps(out, indent=1))
for t, r in out.items():
    print(t, r['selected'], f"NS {r['acc_NS']:.1f} shared {r['acc_shared']:.1f} disagree {r['disagree']} harmed {r['harmed']} rescued {r['rescued']}")
    print('   oracle route:', {k: round(v, 3) for k, v in r['oracle_route'].items()})
    for f, v in r['feature_auc'].items(): print(f'   {f:42s} AUC disagree {v["disagree"] and round(v["disagree"], 3)}  harmed {v["harmed"] and round(v["harmed"], 3)}')
