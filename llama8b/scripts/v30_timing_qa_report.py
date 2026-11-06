"""v30 QuALITY cold timing report: per article mean TTFT per arm (valid blocks only), ratio vs NSP4_25 and vs NS25,
geometric mean over articles of per-article ratios; writes results/v30/timing_qa.json."""
import json, math, statistics as st
from pathlib import Path
R = Path('/home/panzihang/src/prism_layerkv'); T = R / 'studies/v30/timing_qa'
blocks = {}
for b in map(json.loads, (T / 'blocks.jsonl').read_text().splitlines()):
    if b['flag'] or not b['all_ok']: continue
    for r in b['runs']:
        recs = [json.loads(l) for l in (T / 'runs' / r['name'] / 'scored_records.jsonl').read_text().splitlines()]
        assert len({x['uid'] for x in recs}) == len(recs)
        blocks.setdefault(b['cell'], {})[r['arm']] = (st.mean(x['ttft_ms'] for x in recs), st.mean(x['physical_prefetch_kv_bytes'] for x in recs) / 1e6)
arms = sorted({a for c in blocks.values() for a in c})
out = {}
for a in arms:
    cells = [c for c in blocks if a in blocks[c]]
    def g(base):
        xs = [blocks[c][a][0] / blocks[c][base][0] for c in cells if base in blocks[c]]
        return 100 * (math.exp(sum(map(math.log, xs)) / len(xs)) - 1), sum(x < 1 for x in xs), len(xs)
    out[a] = dict(articles=len(cells), ttft_ms=st.mean(blocks[c][a][0] for c in cells), MB=st.mean(blocks[c][a][1] for c in cells),
                  vs_NSP4_25=g('NSP4_25'), vs_NS25=g('NS25'))
(R / 'results/v30/timing_qa.json').write_text(json.dumps(dict(valid_articles=len(blocks), arms=out), indent=1))
print('valid articles', len(blocks))
for a, v in out.items():
    print(f"{a:14s} n={v['articles']:2d} ttft {v['ttft_ms']:7.1f} ms  MB {v['MB']:6.1f}  vsNSP4_25 {v['vs_NSP4_25'][0]:+6.1f}% ({v['vs_NSP4_25'][1]}/{v['vs_NSP4_25'][2]} faster)  vsNS25(P8) {v['vs_NS25'][0]:+6.1f}%")
