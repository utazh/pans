"""a100 分支（7B worktree）：L=28 时把“几何规则必须等于 g16”的断言改为记录差异的警告。
原因：v29m 规则要求消费层与源层选中块完全相同；7B SST-2 25% 下个别层块数因预算取整而不同，
按同一规则跳过这些层即可（14B/32B/Llama 同样处理）。已有 7B 结果不受影响（只影响新运行）。"""
import subprocess, sys
from pathlib import Path
S = Path(sys.argv[1]); p = S / 'scripts/v29m_pipeline.py'; t = p.read_text()
old = "        assert sources == json.loads((ROOT / 'configs/v25_explore/g16.json').read_text())['sources'], 'geometry rule != 7B g16'\n"
new = ("        g16 = json.loads((ROOT / 'configs/v25_explore/g16.json').read_text())['sources']\n"
       "        if sources != g16: print('GEOMETRY differs from 7B g16 (selection-incompatible layers kept native):',\n"
       "                                 [l for l in range(L) if sources[l] != g16[l]], flush=True)\n")
assert t.count(old) == 1; p.write_text(t.replace(old, new))
subprocess.run(['git', '-c', 'user.email=a100@local', '-c', 'user.name=a100', 'commit', '-q', 'scripts/v29m_pipeline.py', '-m',
                '7B: warn instead of assert when selection-compatible geometry differs from g16'], cwd=S, check=True)
print('PATCHED')
