"""生成 Llama 专用的数据副本（原文件只读，不改）：同一套 few-shot 示例与问题文本，改用纯续写格式（不套对话模板）。
前缀 = <|begin_of_text|> + 原示例文本；问题去掉末尾空行、停在 'Sentiment:' 等提示词；标签前缀 ' '（与示例 'Sentiment: positive' 一致）。
依据（只用 fit 校准集、稠密完整 KV）：Llama-3.1-Instruct 在对话模板下倾向先输出 "Here/Based..." 开场白，标签打分接近随机
（SST-2/SUBJ/TREC/RTE 21/16/2/14 of 32），纯续写为 29/31/26/30 of 32（analysis/l8_format_check.py）。
各 runner 均以 add_special_tokens=False 分词，故显式写入 <|begin_of_text|>。
用法: convert_llama.py <src_dir> <dst_dir>   （目录树整体复制；*.jsonl 逐条转换；其他文件原样复制）"""
import json, shutil, sys
from pathlib import Path
QU, QA = '<|im_start|>user\n', '<|im_end|>\n<|im_start|>assistant\n'
LU = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
LA = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

def conv(r):
    p, q = r['prefix_text'], r['query_text']
    assert p.startswith(QU) and p.count('<|im_') == 1, r['uid']
    assert q.endswith(QA) and q.count('<|im_') == 2, r['uid']
    assert r.get('label_continuation_prefix', '') == '', r['uid']
    r['prefix_text'] = '<|begin_of_text|>' + p[len(QU):]
    r['query_text'] = q[:-len(QA)].rstrip('\n')
    r['label_continuation_prefix'] = ' '
    for k, v in r.items():
        assert not (isinstance(v, str) and '<|im_' in v), (r['uid'], k)
    return r

src, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
assert not dst.exists(), dst
n_files = n_rec = 0
for f in sorted(src.rglob('*')):
    out = dst / f.relative_to(src)
    if f.is_dir():
        out.mkdir(parents=True, exist_ok=True); continue
    out.parent.mkdir(parents=True, exist_ok=True)
    if f.suffix == '.jsonl':
        lines = [json.dumps(conv(json.loads(l)), ensure_ascii=False) for l in f.read_text(encoding='utf-8').splitlines() if l.strip()]
        out.write_text(''.join(l + '\n' for l in lines), encoding='utf-8'); n_files += 1; n_rec += len(lines)
    else:
        assert '<|im_' not in f.read_bytes().decode('utf-8', 'replace'), f
        shutil.copy2(f, out)
print('converted', src, '->', dst, 'jsonl files', n_files, 'records', n_rec)
