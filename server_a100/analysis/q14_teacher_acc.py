"""Llama 模板转换的合理性检查：teacher（不共享，P8，25%）在 fit/val 校准问题上的准确率。"""
import json, sys
sys.path.insert(0, '/home/shensicheng/qyd/pans/src_q14/src')
from pathlib import Path
from transformers import AutoTokenizer
from contiguous_fuxian.sharedkv_v24.records import load
B = Path('/home/shensicheng/qyd/pans'); W = B / 'src_q14/studies/v29m_s800'
tok = AutoTokenizer.from_pretrained(B / 'models/Qwen2.5-14B-Instruct')
for cell in ('sst2_b25', 'subj_b25', 'trec_b25', 'rte_b25'):
    task = cell.split('_b')[0]
    for split, role, kind in (('fit', 'fit', 'fit'), ('val', 'validation', 'val')):
        d = W / 'teacher' / f'{cell}_{split}'
        if not d.exists(): continue
        recs = load(d, task, role)
        data = B / 'src/data/v26/p0' / (('canonical_' if task == 'subj' else '') + kind) / f'{task}.jsonl'
        ans = {tuple(tok(r['query_text'], add_special_tokens=False).input_ids): r['answer'] for r in map(json.loads, data.read_text().splitlines())}
        ok = n = miss = 0
        for r in recs.values():
            a = ans.get(tuple(r['query_token_ids']))
            if a is None: miss += 1; continue
            sc = r['teacher_scores']; lab = r['label_order']
            pred = max(sc, key=sc.get) if isinstance(sc, dict) else lab[max(range(len(lab)), key=lambda i: sc[i])]
            ok += pred == a; n += 1
        print(f'{cell} {split}: acc {100 * ok / max(n, 1):.1f}% ({ok}/{n}), unmatched {miss}, labels {lab}', flush=True)
