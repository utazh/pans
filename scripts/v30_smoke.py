"""v30 smoke/parity: the v30 launcher at budget 25 must reproduce the unchanged chain exactly (same predictions, label
log-probs, physical bytes) on 16 TREC dev queries; then one fractional-budget run. Usage: v30_smoke.py <gpu>"""
import json, runpy, sys
from pathlib import Path
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R, PY = C['R'], C['PY']; g = sys.argv[1]
OUT = R / 'studies/v30/smoke'; OUT.mkdir(parents=True, exist_ok=True)
dev = C['DEV']; pc = C['PCACHE'][g]
def job(name, budget, ext_needed):
    o = OUT / 'runs' / name
    return dict(name=name, argv=[PY, C['entry'](ext_needed), *C['chain'](o, task='trec', budget=budget, period=8, samples=16, bundle=dev, pcache=pc)])
C['run_queues'](OUT, {g: [job('base_b25', 25, False), job('ext_b25', 25, True), job('ext_b14p2857', 100 * 4 / 28, True)]})
a, ta = C['load_run'](OUT / 'runs/base_b25'); b, tb = C['load_run'](OUT / 'runs/ext_b25'); c, tc = C['load_run'](OUT / 'runs/ext_b14p2857')
same = all(a[u]['prediction'] == b[u]['prediction'] and a[u]['label_candidate_mean_logprobs'] == b[u]['label_candidate_mean_logprobs'] for u in a)
print(json.dumps(dict(parity_predictions_and_logprobs=same, bytes_base=ta['mean_physical_prefetch_kv_bytes'], bytes_ext=tb['mean_physical_prefetch_kv_bytes'],
                      bytes_frac=tc['mean_physical_prefetch_kv_bytes'], frac_over_25=tc['mean_physical_prefetch_kv_bytes'] / ta['mean_physical_prefetch_kv_bytes'],
                      acc=[ta['accuracy'], tb['accuracy'], tc['accuracy']]), indent=1))
