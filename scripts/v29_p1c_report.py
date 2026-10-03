"""Block-paired analysis of a 4-arm v29 timing study (timing_p1c or timing_p1ab): every arm vs every other arm
measured in the SAME blocks. Usage: v29_p1c_report.py <study_dir_name> [ref_arm]"""
from pathlib import Path
import json, random, statistics as st, sys
R = Path('/home/panzihang/src/prism_layerkv'); T = R / 'studies/v29' / sys.argv[1]
blocks = [json.loads(l) for l in (T / 'blocks.jsonl').read_text().splitlines()]
print('blocks', len(blocks), 'flagged', sum(bool(b['flag']) for b in blocks), 'failed', sum(not b['all_ok'] for b in blocks))
data = {}; cold_bad = 0
for b in blocks:
    if b['flag'] or not b['all_ok']: continue
    for r in b['runs']:
        run = T / 'runs' / r['name']
        t = [json.loads(l)['ttft_ms'] for l in (run / 'scored_records.jsonl').read_text().splitlines()]; assert len(t) == 16
        data.setdefault((b['cell'], b['cond']), {}).setdefault(r['arm'], {})[b['block']] = st.mean(t)
        p = Path(f'{run}.cold.jsonl')
        if p.exists(): cold_bad += sum(json.loads(l).get('status') != 'completed' for l in p.read_text().splitlines())
print('cold requests not completed:', cold_bad)
def ci(xs, n=4000):
    rng = random.Random(0); m = sorted(st.mean(rng.choice(xs) for _ in xs) for _ in range(n)); return m[int(.025 * n)], m[int(.975 * n)]
out = {}
for (cell, cond), arms in sorted(data.items(), key=lambda kv: (kv[0][1] != 'preserve', kv[0][0])):
    names = list(arms); ks = sorted(arms[names[0]])
    print(f"\n{cell} {cond}: " + '  '.join(f"{a} {st.mean(arms[a].values()):.1f} ms" for a in names))
    for a in names:
        for b in names:
            if a == b: continue
            if len(sys.argv) > 2 and a != sys.argv[2] and b not in ('CKV', 'IMP', 'I'): continue
            rel = [arms[a][k] / arms[b][k] - 1 for k in ks]; lo, hi = ci(rel)
            out[f'{cell}|{cond}|{a}_vs_{b}'] = dict(rel_pct=100 * st.mean(rel), ci=[100 * lo, 100 * hi], faster_blocks=sum(x < 0 for x in rel), blocks=len(rel))
            if names.index(a) == 1 or (a == 'I' and b in ('CKV', 'IMP')) or len(sys.argv) > 2:
                print(f"   {a:>8s} vs {b:<8s} {100 * st.mean(rel):+7.1f}%  CI [{100 * lo:+6.1f},{100 * hi:+6.1f}]  faster {sum(x < 0 for x in rel)}/{len(rel)}")
(R / 'results' / f'v29_{sys.argv[1]}.json').write_text(json.dumps(out, indent=1))
