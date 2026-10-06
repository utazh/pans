"""A100 移植的最小、可审计补丁（a100 分支）：
- scripts/run_round4.py: --model-path / --probe-query-heads / --selector-kv-head-ids 可由环境变量
  PANS_MODEL_PATH / PANS_PROBE_QUERY_HEADS / PANS_SELECTOR_KV_HEAD_IDS 覆盖；未设置时与 v4 完全相同。
- scripts/v26_common.py: PY / MODEL 可由 PANS_PY / PANS_MODEL_PATH 覆盖；未设置时与 v4 完全相同。
- 同步更新 sharedkv_v21/guard.py 与 sharedkv_v27/guard.py 中这两个文件的 git blob 哈希（重新审计后的版本）。
先恢复 v4 原文件，再从原文做替换，保证补丁只包含上述改动。"""
import hashlib, subprocess, sys
from pathlib import Path
S = Path('/home/shensicheng/qyd/pans/src')
def blob(p): d = p.read_bytes(); return hashlib.sha1(b'blob ' + str(len(d)).encode() + b'\0' + d).hexdigest()
subprocess.run(['git', 'checkout', 'v4', '--', 'scripts/run_round4.py', 'scripts/v26_common.py',
                'src/contiguous_fuxian/sharedkv_v21/guard.py', 'src/contiguous_fuxian/sharedkv_v27/guard.py'], cwd=S, check=True)
r4 = S / 'scripts/run_round4.py'; vc = S / 'scripts/v26_common.py'
old_r4, old_vc = blob(r4), blob(vc)
t = r4.read_text()
reps = [("'--model-path','/data1/llm/Qwen/Qwen2.5-7B-Instruct',",
         "'--model-path',os.environ.get('PANS_MODEL_PATH','/data1/llm/Qwen/Qwen2.5-7B-Instruct'),"),
        ("'--online-selection','--probe-query-heads','0,7,14,21','--selector-kv-head-ids','0,1,2,3',",
         "'--online-selection','--probe-query-heads',os.environ.get('PANS_PROBE_QUERY_HEADS','0,7,14,21'),'--selector-kv-head-ids',os.environ.get('PANS_SELECTOR_KV_HEAD_IDS','0,1,2,3'),")]
for a, b in reps:
    assert t.count(a) == 1, a; t = t.replace(a, b)
r4.write_text(t)
t = vc.read_text()
for a, b in [("PY='/home/panzihang/venvs/vllm-stable/bin/python'", "PY=os.environ.get('PANS_PY','/home/panzihang/venvs/vllm-stable/bin/python')"),
             ("MODEL='/data1/llm/Qwen/Qwen2.5-7B-Instruct'", "MODEL=os.environ.get('PANS_MODEL_PATH','/data1/llm/Qwen/Qwen2.5-7B-Instruct')")]:
    assert t.count(a) == 1, a; t = t.replace(a, b)
vc.write_text(t)
new_r4, new_vc = blob(r4), blob(vc)
for g in ['src/contiguous_fuxian/sharedkv_v21/guard.py', 'src/contiguous_fuxian/sharedkv_v27/guard.py']:
    p = S / g; t = p.read_text()
    assert old_r4 in t; t = t.replace(old_r4, new_r4)
    if old_vc in t: t = t.replace(old_vc, new_vc)
    p.write_text(t)
print('run_round4', old_r4, '->', new_r4); print('v26_common', old_vc, '->', new_vc)
subprocess.run(['git', 'diff', '--stat', 'v4'], cwd=S)
subprocess.run(['git', '-c', 'user.email=a100@local', '-c', 'user.name=a100', 'commit', '-qam',
                'A100 port: env-overridable model path / GQA head ids (defaults unchanged); guard hashes re-audited'], cwd=S)
print('PATCH_DONE')
