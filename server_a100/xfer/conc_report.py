"""P2 报告：稳态窗口 = [所有 worker 都已完成第 1 个请求, 第一个 worker 完成最后一个请求]。
窗口内完成的请求：吞吐（条/秒，全部 worker 合计）、TTFT 均值/p50/p95、sda 读带宽；冷认证计数用全部请求。
用法: conc_report.py <study_dir> > out.json"""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
D = Path(sys.argv[1]); plan = json.loads((D / 'plan.json').read_text())

def pct(xs, q):
    xs = sorted(xs); k = (len(xs) - 1) * q; f = int(k); return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)

rows = []
for cf in sorted((D / 'configs').glob('*.json')):
    c = json.loads(cf.read_text()); n = c['n']; name = cf.stem
    if any(c['returncodes']): rows.append(dict(name=name, error=c['returncodes'])); continue
    done_t = defaultdict(dict); disk = []
    for e in c['events']:
        if 'w' in e:
            for i in range(1, e['done'] + 1): done_t[e['w']].setdefault(i, e['t'])
        else: disk.append((e['t'], e['sda_read']))
    ttft = {w: [json.loads(l)['ttft_ms'] for l in (D / 'runs' / f'{name}_w{w}' / 'scored_records.jsonl').read_text().splitlines()] for w in range(n)}
    cold_ok = cold_n = 0
    for w in range(n):
        for l in (D / 'runs' / f'{name}_w{w}.cold.jsonl').read_text().splitlines():
            x = json.loads(l); cold_n += 1; cold_ok += x.get('status') == 'completed' and x.get('application_payload_empty', False)
    t0 = max(done_t[w][1] for w in range(n)); t1 = min(done_t[w][len(ttft[w])] for w in range(n))
    inwin = [ttft[w][i - 1] for w in range(n) for i, t in done_t[w].items() if t0 < t <= t1]
    span = t1 - t0
    def disk_at(t): return min(disk, key=lambda d: abs(d[0] - t))[1]
    rows.append(dict(name=name, cell=c['cell'], arm=c['arm'], n=n, rep=c['rep'], window_s=round(span, 1), n_window=len(inwin),
                     throughput_rps=len(inwin) / span if span > 0 else None,
                     ttft_mean=st.mean(inwin) if inwin else None, ttft_p50=pct(inwin, .5) if inwin else None,
                     ttft_p95=pct(inwin, .95) if inwin else None,
                     sda_read_MBps=(disk_at(t1) - disk_at(t0)) / span / 1e6 if span > 0 else None,
                     ttft_all_mean=st.mean(x for v in ttft.values() for x in v), cold=dict(ok=cold_ok, total=cold_n)))
agg = defaultdict(list)
for r in rows:
    if 'error' not in r: agg[(r['cell'], r['n'], r['arm'])].append(r)
summary = {}
for (cell, n, arm), rs in sorted(agg.items()):
    m = lambda k: st.mean(r[k] for r in rs if r[k] is not None)
    summary[f'{cell}|n{n}|{arm}'] = dict(reps=len(rs), throughput_rps=m('throughput_rps'), ttft_mean=m('ttft_mean'), ttft_p50=m('ttft_p50'),
                                         ttft_p95=m('ttft_p95'), sda_read_MBps=m('sda_read_MBps'), n_window=sum(r['n_window'] for r in rs),
                                         cold_ok=sum(r['cold']['ok'] for r in rs), cold_total=sum(r['cold']['total'] for r in rs))
for k, v in summary.items():
    print(f"{k:22s} reps={v['reps']} thr={v['throughput_rps']:.3f}/s ttft mean={v['ttft_mean']:.0f} p50={v['ttft_p50']:.0f} p95={v['ttft_p95']:.0f} ms "
          f"sda={v['sda_read_MBps']:.0f} MB/s win_n={v['n_window']} cold={v['cold_ok']}/{v['cold_total']}", file=sys.stderr)
print(json.dumps(dict(rows=rows, summary=summary), indent=1))
