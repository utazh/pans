"""G2 校准数据：通用文本（WikiText-2 train）切成若干段，每段从位置 0 开始做一次完整前向，
把每层 K/V（HF 缓存里的 post-RoPE 张量）用与前缀存储相同的函数写盘。与任务数据完全无关。
用法: mlr_wiki_kv.py <worktree> <model_dir> <parquet> <out_root> [nseg=8] [seglen=4096]"""
import json, sys, torch
from pathlib import Path
root, model_dir, parquet, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
nseg = int(sys.argv[5]) if len(sys.argv) > 5 else 8; seglen = int(sys.argv[6]) if len(sys.argv) > 6 else 4096
sys.path.insert(0, str(root / 'src'))
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from contiguous_fuxian.sparse_qwen_reprefill import _write_prefix_tensor, _tensor_path

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
rows = [t for t in pd.read_parquet(parquet)['text'].tolist() if t.strip()]
ids = tok(''.join(rows), add_special_tokens=False).input_ids
bos = [tok.bos_token_id] if model.config.model_type == 'llama' else []      # 与前缀存储一致：Llama 前缀以 <|begin_of_text|> 开头
need = nseg * (seglen - len(bos)); assert len(ids) >= need, (len(ids), need)
for i in range(nseg):
    seg = bos + ids[i * (seglen - len(bos)):(i + 1) * (seglen - len(bos))]
    with torch.no_grad():
        cache = model(torch.tensor([seg]).cuda(), use_cache=True).past_key_values
    task = f'wiki{i}'; (out / task).mkdir(parents=True, exist_ok=True)
    for l, layer in enumerate(cache.layers):
        _write_prefix_tensor(_tensor_path(out, task, l, 'key'), layer.keys)
        _write_prefix_tensor(_tensor_path(out, task, l, 'value'), layer.values)
    k0 = cache.layers[0].keys
    (out / task / 'metadata.json').write_text(json.dumps(dict(task=task, prefix_tokens=len(seg), kv_heads=int(k0.shape[1]),
        head_dim=int(k0.shape[-1]), layers=len(cache.layers), source='wikitext-2-raw-v1 train', layout='token,kv_head,head_dim',
        dtype='bfloat16'), indent=2))
    print('wrote', task, len(seg), flush=True)
print('WIKI_KV_DONE', nseg * seglen, 'tokens')
