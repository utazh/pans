"""a100-l8 分支：geometry_map 增加回退。相邻三元组 (p+1,p+2,p+3)/(p+5,p+6,p+7) 的消费层数不足目标时
（Llama-3.1 25%：精确块预算让相邻层 60/59 块交替，相邻层选择永不相同），改用隔层三元组
(p+2,p+4,p+6)/(p+3,p+5,p+7)，其余规则不变（中间层为源、深层优先、消费层必须与源层选择完全相同、目标 2*round(3L/14)）。
7B（L=28）相邻规则已达目标，结果不变（原有 g16 断言保留）。"""
import subprocess, sys
from pathlib import Path
S = Path(sys.argv[1]); p = S / 'scripts/v29m_pipeline.py'; t = p.read_text()
old = """    cands = []
    for s in range(0, L, period):
        for g in ((s + 1, s + 2, s + 3), (s + 5, s + 6, s + 7)):
            if g[-1] < min(s + period, L): cands.append(g)
    target = 2 * round(L * frac / 2); src = list(range(L)); got = 0
    same = (lambda x, y: all(r['selection_by_layer'][x] == r['selection_by_layer'][y] for r in records)) if records else (lambda x, y: True)
    for g in reversed(cands):
        if got >= target: break
        for c in (g[2], g[0]):
            if got < target and same(c, g[1]): src[c] = g[1]; got += 1
    return src
"""
new = """    target = 2 * round(L * frac / 2)
    same = (lambda x, y: all(r['selection_by_layer'][x] == r['selection_by_layer'][y] for r in records)) if records else (lambda x, y: True)
    def build(shape):
        cands = []
        for s in range(0, L, period):
            for g in shape(s):
                if g[-1] < min(s + period, L): cands.append(g)
        src = list(range(L)); got = 0
        for g in reversed(cands):
            if got >= target: break
            for c in (g[2], g[0]):
                if got < target and same(c, g[1]): src[c] = g[1]; got += 1
        return src, got
    src, got = build(lambda s: ((s + 1, s + 2, s + 3), (s + 5, s + 6, s + 7)))
    if got < target:   # 预算取整使相邻层块数交替（Llama-3.1 25%: 60/59）→ 隔层三元组，组内块数相同
        src2, got2 = build(lambda s: ((s + 2, s + 4, s + 6), (s + 3, s + 5, s + 7)))
        if got2 > got: print('GEOMETRY stride-2 fallback', got, '->', got2, flush=True); src = src2
    return src
"""
assert t.count(old) == 1; p.write_text(t.replace(old, new))
subprocess.run(['git', '-c', 'user.email=a100@local', '-c', 'user.name=a100', 'commit', '-qam',
                'l8: geometry_map stride-2 fallback when adjacent groups are budget-incompatible (7B unchanged)'], cwd=S, check=True)
print('GEOMETRY_PATCH_DONE')
