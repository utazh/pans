import sys
sys.path.insert(0, '/home/shensicheng/qyd/pans/src_l8/src')
from pathlib import Path
from contiguous_fuxian.sharedkv_v24.records import load
W = Path('/home/shensicheng/qyd/pans/src_l8/studies/v29m_s800')
cell = sys.argv[1] if len(sys.argv) > 1 else 'sst2_b25'; task = cell.split('_b')[0]
fit = load(W / 'teacher' / f'{cell}_fit', task, 'fit')
recs = list(fit.values()); r = recs[0]; sel = r['selection_by_layer']; L = len(sel)
print('layers', L, 'type', type(sel[0]).__name__)
def desc(x):
    try: return f'len={len(x)} first={list(x)[:4]}'
    except TypeError: return repr(x)[:80]
for l in range(L): print(l, desc(sel[l]))
for s in range(0, L, 8):
    eq = [[all(q['selection_by_layer'][a] == q['selection_by_layer'][b] for q in recs) for b in range(s, s + 8)] for a in range(s, s + 8)]
    print('period', s, 'identical-to-leader:', [int(e) for e in eq[0]], ' pairs(1,2)(3,2)(5,6)(7,6):',
          [int(eq[1][2]), int(eq[3][2]), int(eq[5][6]), int(eq[7][6])])
print('keys', sorted(k for k in r.keys())[:40])
