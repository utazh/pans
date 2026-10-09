"""Llama prompt 格式选择（只用 fit 校准集，稠密完整 KV，打分与系统一致：标签 token 平均对数概率）。
V1 chat：现有转换（user 轮放示例+问题，assistant 头后直接接标签）。
V2 plain：纯续写，<|begin_of_text|> + 原示例文本；问题去掉末尾空行停在 'Sentiment:'，标签前缀 ' '（与示例 'Sentiment: positive' 一致）。"""
import json, sys, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
B = Path('/home/shensicheng/qyd/pans'); M = B / 'models/Llama-3.1-8B-Instruct'
QU, QA = '<|im_start|>user\n', '<|im_end|>\n<|im_start|>assistant\n'
LU = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'; LA = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 32

def variants(r):
    body, q = r['prefix_text'][len(QU):], r['query_text'][:-len(QA)]
    return {'V1_chat': (LU + body, q + LA, ''), 'V2_plain': ('<|begin_of_text|>' + body, q.rstrip('\n'), ' ')}

@torch.no_grad()
def score(prefix, query, cp, labels):
    ctx = tok(prefix + query, add_special_tokens=False).input_ids; out = {}
    for lab in labels:
        lt = tok(cp + lab, add_special_tokens=False).input_ids
        lp = model(torch.tensor([ctx + lt]).cuda()).logits[0, len(ctx) - 1:-1].float().log_softmax(-1)
        out[lab] = sum(lp[i, t].item() for i, t in enumerate(lt)) / len(lt)
    return out

for task in ('sst2', 'subj', 'trec', 'rte'):
    path = B / f'src/data/v26/p0/{"canonical_" if task == "subj" else ""}fit/{task}.jsonl'
    rows = [json.loads(l) for l in path.read_text().splitlines()][:N]
    res = {}
    for name in ('V1_chat', 'V2_plain'):
        ok = 0
        for r in rows:
            p, q, cp = variants(r)[name]; s = score(p, q, cp, r['labels'])
            ok += max(s, key=s.get) == r['answer']
        res[name] = f'{ok}/{len(rows)}'
    print(task, res, flush=True)
