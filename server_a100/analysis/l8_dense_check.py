"""区分问题来源：纯 HF 稠密前向（完整 KV，不经过我们的系统）在同样的 Llama 模板 prompt 上做标签对数似然打分。"""
import json, sys, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
B = Path('/home/shensicheng/qyd/pans'); M = B / 'models/Llama-3.1-8B-Instruct'
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16).cuda().eval()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
for task in ('sst2', 'trec', 'rte'):
    rows = [json.loads(l) for l in (B / f'work/l8/data/v26/p0/fit/{task}.jsonl').read_text().splitlines()][:N]
    labels = sorted({r['answer'] for r in [json.loads(l) for l in (B / f'work/l8/data/v26/p0/fit/{task}.jsonl').read_text().splitlines()]})
    ids0 = tok(rows[0]['prefix_text'] + rows[0]['query_text'], add_special_tokens=False).input_ids
    print(task, 'prompt tokens', len(ids0), 'first ids', ids0[:6], tok.convert_ids_to_tokens(ids0[:6]), 'tail', tok.convert_ids_to_tokens(ids0[-8:]), flush=True)
    ok = 0
    with torch.no_grad():
        for r in rows:
            ctx = tok(r['prefix_text'] + r['query_text'], add_special_tokens=False).input_ids
            out = model(torch.tensor([ctx]).cuda(), use_cache=True); pkv = out.past_key_values; last = out.logits[0, -1].float().log_softmax(-1)
            scores = {}
            for lab in labels:
                lt = tok(lab, add_special_tokens=False).input_ids
                s = last[lt[0]].item()
                if len(lt) > 1:
                    import copy
                    o2 = model(torch.tensor([lt[:-1]]).cuda(), past_key_values=copy.deepcopy(pkv), use_cache=True)
                    lp = o2.logits[0].float().log_softmax(-1); s += sum(lp[i, lt[i + 1]].item() for i in range(len(lt) - 1))
                scores[lab] = s
            pred = max(scores, key=scores.get); ok += pred == r['answer']
        top = tok.convert_ids_to_tokens(last.topk(5).indices.tolist())
    print(f'{task}: dense full-KV label-loglik acc {ok}/{len(rows)}; last-row top5 next tokens {top}', flush=True)
