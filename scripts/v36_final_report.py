"""v36 final tables: Qwen2.5-7B and Llama-3.1-8B (4 tasks, 25 %) and QuALITY (7B), every method, accuracy (frozen test sets),
restored MB/request, SSD-cold and warm TTFT (same blocks, valid blocks only; a rerun of a flagged block replaces it), and
block-paired geometric-mean slowdown of each method relative to RAPS-P4. Writes results/v36/final.json + final.csv."""
import csv, json, math, os, statistics as st
from pathlib import Path
Q7 = Path('/home/panzihang/src/prism_layerkv'); L8 = Path('/data1/pans_l8/raps_l8')
TASKS = ['sst2', 'subj', 'trec', 'rte']

def load_acc(p):
    p = Path(p); rows = [json.loads(l) for l in (p / 'scored_records.jsonl').read_text().splitlines()]
    t = list(json.loads((p / 'summary.json').read_text())['tasks'].values())[0]
    return rows, t

def timing(dirs):
    """{(cell, cond): {block: {arm: ttft}}} from valid blocks; later dirs / rerun suffixes override earlier ones."""
    out = {}
    for d in dirs:
        d = Path(d)
        if not (d / 'blocks.jsonl').exists(): continue
        for b in map(json.loads, (d / 'blocks.jsonl').read_text().splitlines()):
            if b['flag'] or not b['all_ok']: continue
            arms = {}
            for r in b['runs']:
                recs = [json.loads(l) for l in (d / 'runs' / r['name'] / 'scored_records.jsonl').read_text().splitlines()]
                assert len({x['uid'] for x in recs}) == len(recs)
                arms[r['arm']] = dict(ttft=st.mean(x['ttft_ms'] for x in recs), gpu_hit=st.mean(x.get('prefetch_gpu_source_fraction', 0) for x in recs))
            out.setdefault((b['cell'], b['cond']), {})[b['block']] = arms
    return out

def summarize(tim, cell, cond, arm):
    bl = tim.get((cell, cond), {}); ks = [k for k in bl if arm in bl[k] and 'RAPS4' in bl[k]]
    if not ks: return None
    xs = [bl[k][arm]['ttft'] / bl[k]['RAPS4']['ttft'] for k in ks]
    return dict(ttft=round(st.mean(bl[k][arm]['ttft'] for k in ks), 1), blocks=len(ks), x_of_raps=round(math.exp(sum(map(math.log, xs)) / len(xs)), 2),
                raps_faster=f'{sum(x > 1 for x in xs)}/{len(xs)}' if arm != 'RAPS4' else '', gpu_hit=round(st.mean(bl[k][arm]['gpu_hit'] for k in ks), 2))

ARMS = ['RAPS4', 'NS4', 'NS8', 'CKV', 'IMP', 'AS', 'CAKE', 'LAVA']
result = {}
# ---- Qwen2.5-7B ----
sel7 = {c[:-4]: v['selected'] for c, v in json.loads((Q7 / 'results/v30/p4_selection.json').read_text())['cells'].items()}
tim7 = timing([Q7 / 'studies/v33/timing', Q7 / 'studies/v33/timing_fix'])
def acc7(t, arm):
    q = Q7 / 'studies/v30/quality/runs'
    p = {'NS8': Q7 / f'studies/v29/test_quality/runs/test_I_{t}_b25', 'NS4': q / f'ns_p4_test_{t}_k25',
         'RAPS4': q / (f'p4_test_{sel7[t]}_{t}_k25' if sel7[t] != 'I' else f'ns_p4_test_{t}_k25'),
         'CKV': Q7 / f'studies/v29/c_quality/runs/contigkv_{t}_b25', 'IMP': Q7 / f'studies/v29/c_quality/runs/impress_{t}_b25',
         'AS': Q7 / f'studies/v33/as/runs/as_test_{t}', 'CAKE': q / f'cake_static_{t}_k25', 'LAVA': q / f'lava_static_{t}_k25'}[arm]
    _, s = load_acc(p); return round(100 * s['accuracy'], 2), round(s['mean_physical_prefetch_kv_bytes'] / 1e6, 1)
# ---- Llama-3.1-8B ----
sel8 = {c[:-4]: v['selected'] for c, v in json.loads((L8 / 'results/v30/p4_selection.json').read_text())['cells'].items()}
tim8 = timing([L8 / 'studies/v34/l8_timing', L8 / 'studies/v34/l8_timing_fix'])
def acc8(t, arm):
    q = L8 / 'studies/v30/quality/runs'; b = L8 / 'studies/v34/baselines/runs'
    p = {'NS8': q / f'ns_p8_test_{t}_k25', 'NS4': q / f'ns_p4_test_{t}_k25',
         'RAPS4': q / (f'p4_test_{sel8[t]}_{t}_k25' if sel8[t] != 'I' else f'ns_p4_test_{t}_k25'),
         'CKV': b / f'CKV_test_{t}', 'IMP': b / f'IMP_test_{t}', 'AS': b / f'AS_test_{t}',
         'CAKE': q / f'cake_static_{t}_k25', 'LAVA': q / f'lava_static_{t}_k25'}[arm]
    _, s = load_acc(p); return round(100 * s['accuracy'], 2), round(s['mean_physical_prefetch_kv_bytes'] / 1e6, 1)
for model, accf, tim, sel in (('qwen7b', acc7, tim7, sel7), ('llama8b', acc8, tim8, sel8)):
    for t in TASKS:
        for arm in ARMS:
            a, mb = accf(t, arm)
            result[f'{model}|{t}|{arm}'] = dict(model=model, task=t, arm=arm, selected=sel[t] if arm == 'RAPS4' else '', acc=a, MB=mb,
                                               cold=summarize(tim, f'{t}_b25', 'preserve', arm), warm=summarize(tim, f'{t}_b25', 'warm', arm))
# ---- QuALITY (7B, 12 articles; accuracy pooled over all 224 questions, TTFT per article block then averaged) ----
timq = timing([Q7 / 'studies/v35/timing_qa'])
QV29 = lambda s, n, slot: Q7 / f'studies/v29/quality/{s}/eval/{n}_{slot}'
def qa_dir(arm, s, slot):
    return {'NS8': QV29(s, 'I', slot), 'NS4': Q7 / f'studies/v30/quality_qa/runs/{s}_{slot}_NSP4_25',
            'RAPS4': Q7 / f'studies/v30/qa_p4/{s}_{slot}/eval/RAPSP4_G24', 'CKV': Q7 / f'studies/v35/qa_baselines/runs/{s}_{slot}_CKV',
            'IMP': Q7 / f'studies/v35/qa_baselines/runs/{s}_{slot}_IMP', 'AS': Q7 / f'studies/v35/qa_baselines/runs/{s}_{slot}_AS',
            'CAKE': Q7 / f'studies/v30/quality_qa/runs/{s}_{slot}_CAKE_25', 'LAVA': Q7 / f'studies/v30/quality_qa/runs/{s}_{slot}_LAVA_25',
            'FULL': QV29(s, 'FULLKV', slot)}[arm]
cells = [f'{s}_{slot}' for s in ('f0', 'f1', 'f2') for slot in TASKS]
for arm in ARMS + ['FULL']:
    c = n = 0; mbs = []
    for cell in cells:
        s, slot = cell.split('_'); rows, sm = load_acc(qa_dir(arm, s, slot)); c += sum(r['correct'] for r in rows); n += len(rows)
        mbs.append((sm.get('mean_physical_prefetch_kv_bytes') or float('nan')) / 1e6)
    def qsum(cond):
        xs, tt, ok = [], [], 0
        for cell in cells:
            bl = timq.get((cell, cond), {}).get(0)
            if bl and arm in bl and 'RAPS4' in bl:
                xs.append(bl[arm]['ttft'] / bl['RAPS4']['ttft']); tt.append(bl[arm]['ttft']); ok += 1
        if not xs: return None
        return dict(ttft=round(st.mean(tt), 1), blocks=ok, x_of_raps=round(math.exp(sum(map(math.log, xs)) / len(xs)), 2),
                    raps_faster=f'{sum(x > 1 for x in xs)}/{len(xs)}' if arm != 'RAPS4' else '')
    result[f'quality|{arm}'] = dict(model='qwen7b', task='quality', arm=arm, acc=round(100 * c / n, 2), n=n, MB=round(st.mean(m for m in mbs if m == m), 1) if any(m == m for m in mbs) else None,
                                    cold=qsum('preserve') if arm != 'FULL' else None, warm=qsum('warm') if arm != 'FULL' else None)
out = Q7 / 'results/v36'; out.mkdir(parents=True, exist_ok=True)
(out / 'final.json').write_text(json.dumps(result, indent=1))
with (out / 'final.csv').open('w', newline='') as f:
    w = csv.writer(f); w.writerow(['model', 'task', 'arm', 'selected', 'acc', 'MB', 'cold_ms', 'cold_blocks', 'cold_x_of_RAPS', 'cold_raps_faster',
                                   'warm_ms', 'warm_blocks', 'warm_x_of_RAPS', 'warm_raps_faster', 'warm_gpu_hit'])
    for v in result.values():
        c, wm = v.get('cold') or {}, v.get('warm') or {}
        w.writerow([v['model'], v['task'], v['arm'], v.get('selected', ''), v['acc'], v['MB'], c.get('ttft'), c.get('blocks'), c.get('x_of_raps'), c.get('raps_faster'),
                    wm.get('ttft'), wm.get('blocks'), wm.get('x_of_raps'), wm.get('raps_faster'), wm.get('gpu_hit')])
print((out / 'final.csv').read_text())
