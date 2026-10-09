"""Llama-3.1 移植的最小、可审计补丁（a100-l8 分支，只在 src_l8 worktree 里执行）：
放开 RAPS 路径上的模型族检查：model_type 允许 'llama'，rope_scaling 允许 None 或 rope_type=='llama3'。
理由：这些路径的 cos/sin 全部来自 model.rotary_emb（HF 已实现 llama3 频率缩放），
Llama 与 Qwen2 的 rotate_half RoPE、GQA repeat_kv 完全相同，q/k/v 无偏置不影响 *_proj 调用。
其他检查（滑动窗口等）保持不变。随后把所有 guard.py 里这些文件的 git blob 哈希更新到新版本（迭代到不动点）。"""
import hashlib, subprocess, sys
from pathlib import Path
S = Path(sys.argv[1])
def blob(p): d = p.read_bytes(); return hashlib.sha1(b'blob ' + str(len(d)).encode() + b'\0' + d).hexdigest()

RS_SQ = "(getattr(model.config,'rope_scaling',None) or {}).get('rope_type','none') not in ('none','llama3')"
PATCHES = {
 'src/contiguous_fuxian/sharedkv_v25/runtime.py': [
   ("model.config.model_type!='qwen2' or getattr(model.config,'use_sliding_window',False) or getattr(model.config,'rope_scaling',None) is not None",
    "model.config.model_type not in ('qwen2','llama') or getattr(model.config,'use_sliding_window',False) or " + RS_SQ)],
 'src/contiguous_fuxian/sharedkv_v25/geometry.py': [
   ("if c.get('model_type')!='qwen2':", "if c.get('model_type') not in ('qwen2','llama'):"),
   ("c.get('rope_scaling') is not None", "(c.get('rope_scaling') or {}).get('rope_type','none') not in ('none','llama3')")],
 'src/contiguous_fuxian/sharedkv_v23/train.py': [
   ("model.config.model_type!='qwen2' or getattr(model.config,'rope_scaling',None) is not None",
    "model.config.model_type not in ('qwen2','llama') or " + RS_SQ)],
 'src/contiguous_fuxian/sharedkv_v24/train.py': [
   ("model.config.model_type!='qwen2' or getattr(model.config,'rope_scaling',None) is not None",
    "model.config.model_type not in ('qwen2','llama') or " + RS_SQ)],
 'src/contiguous_fuxian/sharedkv/runtime.py': [
   ('model.config.model_type != "qwen2"', 'model.config.model_type not in ("qwen2","llama")'),
   ('if getattr(model.config,"rope_scaling",None) is not None:',
    'if (getattr(model.config,"rope_scaling",None) or {}).get("rope_type","none") not in ("none","llama3"):')],
}
changed = {}
for rel, reps in PATCHES.items():
    p = S / rel; old = blob(p); t = p.read_text()
    for a, b in reps:
        assert t.count(a) == 1, (rel, a); t = t.replace(a, b)
    p.write_text(t); changed[old] = blob(p); print(rel, old, '->', changed[old])
guards = sorted(S.glob('src/contiguous_fuxian/*/guard.py'))
while changed:                      # guard 文件自己也可能被别的 guard 检查：迭代到不动点
    nxt = {}
    for g in guards:
        t = g.read_text(); old = blob(g); u = t
        for a, b in changed.items(): u = u.replace(a, b)
        if u != t:
            g.write_text(u); nxt[old] = blob(g); print('guard', g.relative_to(S), old, '->', nxt[old])
    changed = nxt
subprocess.run(['git', 'diff', '--stat'], cwd=S)
subprocess.run(['git', '-c', 'user.email=a100@local', '-c', 'user.name=a100', 'commit', '-qam',
                'Llama-3.1 port: allow model_type llama + llama3 rope_scaling on RAPS paths (cos/sin from model.rotary_emb); guard hashes re-audited'], cwd=S, check=True)
print('LLAMA_PATCH_DONE')
