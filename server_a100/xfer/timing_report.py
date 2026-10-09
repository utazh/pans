"""A100 计时报告（与 3090 v29_timing_report 相同口径）：每块 = 每臂 16 条请求平均 TTFT；配对相对变化 = 100*(arm/I - 1)；
被标记的块用其 _r1 重跑（若重跑也被标记则剔除）；6 块均值 + 4000 次块 bootstrap 95% 区间；冷认证统计。
用法: timing_report.py <study_dir> [cond] > out.json"""
import json, random, statistics as st, sys
from pathlib import Path
ALL = '--all' in sys.argv; argv = [a for a in sys.argv if a != '--all']   # --all：敏感性分析，被标记的块也计入
D = Path(argv[1]); cond_f = argv[2] if len(argv) > 2 else None
blocks = [json.loads(l) for l in (D / 'blocks.jsonl').read_text().splitlines()]
use = {}
for b in blocks:                                   # 每 (cell,cond,block) 取最后一个未被标记且全部成功的记录
    k = (b['cell'], b['cond'], b['block'])
    if (b['flag'] is None or ALL) and b['all_ok']: use[k] = b
    elif k not in use: use.setdefault(('_bad',) + k, b)
def mean_ttft(name):
    return st.mean(json.loads(l)['ttft_ms'] for l in (D / 'runs' / name / 'scored_records.jsonl').read_text().splitlines())
cold_ok = cold_n = 0
res = {}
for (cell, cond, blk), b in sorted((k, v) for k, v in use.items() if k[0] != '_bad'):
    if cond_f and cond != cond_f: continue
    ms = {r['arm']: mean_ttft(r['name']) for r in b['runs']}
    for r in b['runs']:
        cf = D / 'runs' / (r['name'] + '.cold.jsonl')
        if cf.exists():
            for l in cf.read_text().splitlines():
                x = json.loads(l); cold_n += 1; cold_ok += x.get('status') == 'completed' and x.get('application_payload_empty', False)
    res.setdefault((cell, cond), []).append(ms)
out = {}
rng = random.Random(0)
for (cell, cond), bl in res.items():
    arms = [a for a in bl[0] if a != 'I']; rec = dict(blocks=len(bl), I_ms=st.mean(x['I'] for x in bl))
    for a in arms:
        rel = [100 * (x[a] / x['I'] - 1) for x in bl]
        boots = sorted(st.mean(rng.choice(rel) for _ in rel) for _ in range(4000))
        rec[a] = dict(ms=st.mean(x[a] for x in bl), rel_pct=st.mean(rel), ci=[boots[99], boots[3899]], faster=sum(r < 0 for r in rel))
    out[f'{cell}|{cond}'] = rec
bad = [k[1:] for k in use if k[0] == '_bad' and k[1:] not in use]
print(json.dumps(dict(cells=out, cold_checks=dict(ok=cold_ok, total=cold_n), unresolved_flagged_blocks=bad), indent=1))
