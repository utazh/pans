"""读取器来源对比（同一共享图、同一冻结选档规则）：每个 (模型, 单元, 读取器来源) 的开发集净变化、选中档、测试集各档准确率与字节。
来源：s800/v29m=每前缀训练；ridge=每前缀闭式；mlr_wiki=模型级闭式（WikiText）；mlr_loo=模型级闭式（其他任务前缀）；mlt=模型级训练（留一）。"""
import json
from pathlib import Path
B = Path('/home/shensicheng/qyd/pans'); out = {}
SETS = {'l8': ('src_l8', ['s800', 'ridge', 'mlr_wiki', 'mlr_loo', 'mlt']), 'q7': ('src', ['ridge'])}
for model, (wt, tags) in SETS.items():
    for tag in tags:
        W = B / wt / 'studies' / f'v29m_{tag}'
        for cell in ('sst2_b25', 'subj_b25', 'trec_b25', 'rte_b25'):
            sf, tf = W / 'selection' / f'{cell}.json', W / 'test_summary' / f'{cell}.json'
            if not (sf.exists() and tf.exists()): continue
            s, ts = json.loads(sf.read_text()), json.loads(tf.read_text())
            out[f'{model}|{tag}|{cell}'] = dict(selected=s['selected'], dev={k: v['net'] for k, v in s['candidates'].items()},
                                               consumers={k: v['consumers'] for k, v in s['candidates'].items()},
                                               test={k: dict(acc=v['acc'], dpp=v['delta_pp'], bytes=v['bytes'], disagree=v['disagree']) for k, v in ts.items()})
(B / 'analysis/reader_sources.json').write_text(json.dumps(out, indent=1))
for k, v in out.items():
    t = v['test']; sel = v['selected']
    print(f"{k:28s} sel={sel:2s} dev S1/S2/S3={v['dev'].get('S1')}/{v['dev'].get('S2')}/{v['dev'].get('S3')}  "
          f"test dpp S1/S2/S3={t['S1']['dpp']:+.1f}/{t['S2']['dpp']:+.1f}/{t['S3']['dpp']:+.1f}  selected dpp={t[sel]['dpp']:+.2f} bytes {100*(t[sel]['bytes']/t['I']['bytes']-1):+.1f}%")
