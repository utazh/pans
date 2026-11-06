"""v33 report: 7B, 4 tasks x 25 %, every method in the same timing blocks, SSD-cold and warm. Accuracy from the frozen test
sets (NS8: studies/v29/test_quality, NS4/RAPS4/CAKE/LAVA: studies/v30/quality, CKV/IMP: studies/v29/c_quality, AS:
studies/v33/as). TTFT = mean of per-block means over valid (unflagged, all-ok) blocks; ratios are block-paired.
Writes results/v33/table.csv + summary.json. Usage: v33_report.py"""
from pathlib import Path
import csv, json, math, runpy, statistics as st
C = runpy.run_path(str(Path(__file__).with_name('v30_common.py')))
R = C['R']; T = R / 'studies/v33/timing'; OUT = R / 'results/v33'; OUT.mkdir(parents=True, exist_ok=True)
sel = {c: v['selected'] for c, v in json.loads((R / 'results/v30/p4_selection.json').read_text())['cells'].items()}
def acc(arm, t):
    p = {'NS8': R / f'studies/v29/test_quality/runs/test_I_{t}_b25', 'NS4': R / f'studies/v30/quality/runs/ns_p4_test_{t}_k25',
         'RAPS4': R / f'studies/v30/quality/runs/p4_test_{sel[t + "_b25"]}_{t}_k25' if sel[t + '_b25'] != 'I' else R / f'studies/v30/quality/runs/ns_p4_test_{t}_k25',
         'CKV': R / f'studies/v29/c_quality/runs/contigkv_{t}_b25', 'IMP': R / f'studies/v29/c_quality/runs/impress_{t}_b25',
         'AS': R / f'studies/v33/as/runs/as_test_{t}', 'CAKE': R / f'studies/v30/quality/runs/cake_static_{t}_k25',
         'LAVA': R / f'studies/v30/quality/runs/lava_static_{t}_k25'}[arm]
    rows, s = C['load_run'](p); return 100 * s['accuracy'], s['mean_physical_prefetch_kv_bytes'] / 1e6
blocks = {}; bad = []
for b in map(json.loads, (T / 'blocks.jsonl').read_text().splitlines()):
    key = (b['cell'], b['cond'], b['block'])
    if b['flag'] or not b['all_ok']: bad.append(dict(cell=b['cell'], cond=b['cond'], block=b['block'], suffix=b['suffix'], flag=b['flag'], all_ok=b['all_ok'])); continue
    for r in b['runs']:
        recs = [json.loads(l) for l in (T / 'runs' / r['name'] / 'scored_records.jsonl').read_text().splitlines()]
        assert len(recs) == 16 and len({x['uid'] for x in recs}) == 16, r['name']
        gpu = st.mean(x.get('prefetch_gpu_source_fraction', float('nan')) for x in recs); cpu = st.mean(x.get('prefetch_cpu_source_fraction', float('nan')) for x in recs)
        blocks.setdefault(key, {})[r['arm']] = dict(ttft=st.mean(x['ttft_ms'] for x in recs), MB=st.mean(x['physical_prefetch_kv_bytes'] for x in recs) / 1e6,
                                                   ssd_MB=st.mean(x.get('critical_ssd_read_bytes', 0) for x in recs) / 1e6, gpu_hit=gpu, cpu_hit=cpu)
rows = []
for t in C['TASKS']:
    for cond in ('preserve', 'warm'):
        keys = [k for k in blocks if k[0] == f'{t}_b25' and k[1] == cond]
        arms = sorted({a for k in keys for a in blocks[k]})
        for arm in arms:
            ks = [k for k in keys if arm in blocks[k]]
            def ratio(base):
                xs = [blocks[k][arm]['ttft'] / blocks[k][base]['ttft'] for k in ks if base in blocks[k]]
                return (math.exp(sum(map(math.log, xs)) / len(xs)), sum(x < 1 for x in xs), len(xs)) if xs else (None, 0, 0)
            a, mb = acc(arm, t)
            rr = ratio('RAPS4')
            rows.append(dict(task=t, cond=cond, arm=arm, blocks=len(ks), acc=round(a, 2), restored_MB=round(mb, 1),
                             ttft_ms=round(st.mean(blocks[k][arm]['ttft'] for k in ks), 1),
                             speedup_of_RAPS4=None if rr[0] is None else round(rr[0], 2), raps_faster_blocks=f'{rr[2] - rr[1]}/{rr[2]}' if arm != 'RAPS4' else '',
                             gpu_hit=round(st.mean(blocks[k][arm]['gpu_hit'] for k in ks), 3), cpu_hit=round(st.mean(blocks[k][arm]['cpu_hit'] for k in ks), 3),
                             ssd_read_MB=round(st.mean(blocks[k][arm]['ssd_MB'] for k in ks), 1)))
with (OUT / 'table.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(OUT / 'summary.json').write_text(json.dumps(dict(invalid_blocks=bad, valid_blocks=len(blocks), selected=sel), indent=1))
print('valid blocks', len(blocks), 'invalid', len(bad)); [print(b) for b in bad]
for r in rows: print(r)
