"""a100 分支：scripts/run_round4_methods.py（基线最内层入口，不在 guard 列表中）的模型路径与 ContiguousKV 选择 KV 头可由环境变量覆盖。
默认值与 v4 完全相同；IMPRESS 仍只用 KV 头 0（与原复现参数一致）。"""
from pathlib import Path
import subprocess
S = Path('/home/shensicheng/qyd/pans/src'); p = S / 'scripts/run_round4_methods.py'; t = p.read_text()
def rep(a, b):
    global t
    assert t.count(a) == 1, a; t = t.replace(a, b)
rep("argv = ['--model-path', '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct',",
    "argv = ['--model-path', os.environ.get('PANS_MODEL_PATH', '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct'),")
rep("argv += ['--probe-query-heads', '0,1,2', '--selector-kv-head-ids', '0,1,2,3', '--similarity-alpha', '0.6']",
    "argv += ['--probe-query-heads', '0,1,2', '--selector-kv-head-ids', os.environ.get('PANS_SELECTOR_KV_HEAD_IDS', '0,1,2,3'), '--similarity-alpha', '0.6']")
p.write_text(t)
subprocess.run(['git', '-c', 'user.email=a100@local', '-c', 'user.name=a100', 'commit', '-qam',
                'A100: run_round4_methods model path / CKV selector heads via env (defaults unchanged)'], cwd=S, check=True)
print('METHODS_PATCHED')
