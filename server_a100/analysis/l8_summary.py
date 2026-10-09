"""Llama 结果汇总：开发集选档 + 测试集各档（准确率、相对 I 变化、字节、预测改变率）。"""
import json
from pathlib import Path
W = Path('/home/shensicheng/qyd/pans/src_l8/studies/v29m_s800')
out = {}
for cell in ('sst2_b25', 'subj_b25', 'trec_b25', 'rte_b25'):
    sel = json.loads((W / 'selection' / f'{cell}.json').read_text())
    ts = json.loads((W / 'test_summary' / f'{cell}.json').read_text()) if (W / 'test_summary' / f'{cell}.json').exists() else None
    der = json.loads((W / 'done' / f'{cell}_derive').read_text())
    print(f"== {cell}: selected {sel['selected']}  consumers S3={der['consumers']}  dev net/MB:",
          {k: (v['net'], round(v['bytes'] / 1e6, 1)) for k, v in sel['candidates'].items()})
    if ts:
        arms = ts.get('arms', ts)
        for a, r in arms.items():
            if isinstance(r, dict) and 'acc' in r:
                print(f"   test {a:2s} acc {r['acc']:.2f}  d {r['delta_pp']:+.2f}  bytes {r['bytes'] / 1e6:.1f} MB  disagree {r['disagree']:.1f}%")
    out[cell] = dict(selection=sel, test=ts, derive=der)
Path('/home/shensicheng/qyd/pans/analysis/l8_summary.json').write_text(json.dumps(out, indent=1))
