"""Serial-time amortization estimate (GPT definition): N = (added offline build + configuration selection seconds)
/ (same-configuration, same-backend per-request TTFT saving). Not throughput, money or production-benefit guarantees.
build   = teacher capture (fit+val; estimated as 2x the fit->val log-mtime gap) + R4 reader training wall (audit compute.json)
select  = development runs of the shared candidates G16/G20/G24 (the I dev run is counted too: selection needs it)
saving  = I - selected TTFT: pcache SSD-cold (stage B, 16 cells), pcache warm (8 cells), row engine SSD-cold (25% cells).
Writes results/v29_amortization.json."""
import csv, json, os, statistics as st
from pathlib import Path
R = Path('/home/shensicheng/qyd/pans/src')
final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
a1 = {json.loads(l)['name']: json.loads(l)['seconds'] for l in (R / 'studies/v29/a1/status.jsonl').read_text().splitlines()}
a0 = {json.loads(l)['name']: json.loads(l)['seconds'] for l in (R / 'studies/v29/a0/status.jsonl').read_text().splitlines()}
tim = {(r['cell'], r['cond'], r['arm']): r for r in csv.DictReader((R / 'results/v29_timing/table.csv').open())}
row = json.loads((R / 'results/v29_timing_row2.json').read_text()) if (R / 'results/v29_timing_row2.json').exists() else {}
def row_ms(cell, arm):
    d = R / 'studies/v29/timing_row2/runs'
    xs = [json.loads(l)['ttft_ms'] for p in d.glob(f'{cell}_preserve_{arm}_b0*/scored_records.jsonl') for l in p.read_text().splitlines()]
    return st.mean(xs) if xs else None
out = {}
for cell, f in final.items():
    task, b = cell.split('_')
    tag = f'v26_p0_{task}_{b}' + ('_canonical' if task == 'subj' else '')
    comp = R / f'audit/{tag}_compute.json'; proxy = None
    if not comp.exists():  # trec_b25 reader is the historical v25 checkpoint: use the same-task/budget p1 run as proxy
        comp = R / 'audit/v26_p1_trec_b25_compute.json'; proxy = 'v26_p1_trec_b25 (same task/budget, other prefix)'
    train = json.loads(comp.read_text())['wall_seconds_total_including_load']
    lf, lv = R / f'logs/{tag}_capture_fit.log', R / f'logs/{tag}_capture_val.log'
    if not lf.exists(): lf, lv = R / 'logs/v26_p1_trec_b25_capture_fit.log', R / 'logs/v26_p1_trec_b25_capture_val.log'
    capture = 2 * max(0.0, os.path.getmtime(lv) - os.path.getmtime(lf))
    select = sum(a1.get(f'dev_{g}_{cell}', 0) for g in ('G16', 'G20', 'G24')) + a0.get(f'devI_{cell}', 0)
    offline = capture + train + select
    rec = dict(config=f['config'], capture_s=round(capture, 1), train_s=round(train, 1), select_s=round(select, 1),
               offline_s=round(offline, 1), proxy=proxy)
    for cond in ('preserve', 'warm'):
        k = (cell, cond, f['config'] if f['config'] != 'I' else 'I(sel)')
        if k in tim:
            sav = float(tim[k]['I_ms']) - float(tim[k]['arm_ms'])
            rec[f'pcache_{cond}_saving_ms'] = round(sav, 1)
            rec[f'pcache_{cond}_N'] = None if sav <= 0 else round(offline * 1000 / sav)
    if b == 'b25':
        i, s = row_ms(cell, 'I_ROW'), row_ms(cell, 'SEL_ROW' if f['config'] != 'I' else 'I_ROW')
        if i and s:
            sav = i - s; rec['row_cold_saving_ms'] = round(sav, 1); rec['row_cold_N'] = None if sav <= 0 else round(offline * 1000 / sav)
    out[cell] = rec
(R / 'results/v29_amortization.json').write_text(json.dumps(out, indent=1))
print(f"{'cell':9s} cfg  capture  train  select  offline_s | pcache cold save/N | warm save/N | row cold save/N")
for c, r in out.items():
    print(f"{c:9s} {r['config']:4s} {r['capture_s']:7.0f} {r['train_s']:6.0f} {r['select_s']:7.0f} {r['offline_s']:9.0f} | "
          f"{r.get('pcache_preserve_saving_ms','-')!s:>7} / {r.get('pcache_preserve_N','-')!s:>6} | {r.get('pcache_warm_saving_ms','-')!s:>6} / {r.get('pcache_warm_N','-')!s:>6} | "
          f"{r.get('row_cold_saving_ms','-')!s:>6} / {r.get('row_cold_N','-')!s:>6}{'  (train proxy)' if r['proxy'] else ''}")
