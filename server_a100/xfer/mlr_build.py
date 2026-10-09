"""G2 模型级读取器：对每个 (消费层, 源层) 对、每个 KV 头，用同一个闭式带正则最小二乘（affine_ridge，ridge=1e-3，
与训练初值完全相同的函数），但数据换成与目标前缀无关的校准 KV：
  --calib wiki : 通用文本（mlr_wiki_kv.py 生成），与任何任务无关 → 一个模型一套 A/B
  --calib loo  : 其他三个任务的前缀（留一法），目标任务的前缀不参与拟合
共享图、身份字段、S3→S2/S1 回退顺序全部取自 --from 目录（同一流水线的 ridge-only 结果），只替换 A、B、bk、bv。
写出 studies/v29m_<to>/readers/<cell>/{S3,S2,S1}，并复制 capture/train/derive 完成标记，之后用流水线 dev/test 阶段评估。
用法: mlr_build.py <worktree> --from ridge --to mlr_wiki --calib wiki --calib-root <dir> --cells sst2_b25,..."""
import argparse, copy, json, shutil, sys, time
from pathlib import Path
import torch
p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--from', dest='src', required=True); p.add_argument('--to', required=True)
p.add_argument('--calib', choices=('wiki', 'loo'), required=True); p.add_argument('--calib-root'); p.add_argument('--cells', required=True)
p.add_argument('--ridge', type=float, default=1e-3)
a = p.parse_args(); R = Path(a.root); sys.path.insert(0, str(R / 'src'))
from contiguous_fuxian.sharedkv.fit import affine_ridge
from contiguous_fuxian.sharedkv_v21.bundles import isolate
W1, W2 = R / 'studies' / f'v29m_{a.src}', R / 'studies' / f'v29m_{a.to}'
STORE = R / 'runtime_data/prefix_store'; TASKS = ('sst2', 'subj', 'trec', 'rte')

def read(store, task, l, kind):
    m = json.loads((store / task / 'metadata.json').read_text())
    raw = torch.frombuffer(bytearray((store / task / f'layer_{l:02d}_{kind}.bf16').read_bytes()), dtype=torch.uint8)
    return raw.view(torch.bfloat16).reshape(m['prefix_tokens'], m['kv_heads'], m['head_dim']).float()

pool_cache = {}
def pooled(target, l, kind):
    if a.calib == 'wiki':
        store = Path(a.calib_root); names = sorted(d.name for d in store.iterdir() if (d / 'metadata.json').exists())
    else:
        store = STORE; names = [t for t in TASKS if t != target]
    key = (tuple(names), l, kind)
    if key not in pool_cache: pool_cache[key] = torch.cat([read(store, n, l, kind) for n in names])
    return pool_cache[key], names

log = []
for cell in a.cells.split(','):
    task = cell.split('_b')[0]; t0 = time.time()
    bnd = torch.load(W1 / 'readers' / cell / 'S3' / f'{task}.pt', map_location='cpu', weights_only=False)
    new = {}
    for lid, s in enumerate(bnd['sources']):
        if lid == s: continue
        tmpl = bnd['readers'][str(lid)]; parts = {}
        for kind, (wm, wb) in (('key', ('A', 'bk')), ('value', ('B', 'bv'))):
            xs, names = pooled(task, s, kind); xl, _ = pooled(task, lid, kind)
            M, b = zip(*(affine_ridge(xs[:, h], xl[:, h], ridge=a.ridge) for h in range(xs.shape[1])))
            parts[wm] = torch.stack(M); parts[wb] = torch.stack(b)
        r = {k: parts[k].to(torch.bfloat16).to(tmpl[k].dtype) for k in ('A', 'B', 'bk', 'bv')}   # 与导出器相同的 BF16 取整
        assert all(r[k].shape == tmpl[k].shape for k in r), lid
        new[str(lid)] = r
    out = copy.deepcopy(bnd); out['readers'] = new
    out['reader_origin'] = dict(kind=f'model-level ridge ({a.calib})', calibration=names, ridge=a.ridge,
                                note='closed-form; no gradient training; target prefix not used' if a.calib == 'wiki' else 'closed-form; leave-one-task-out prefixes')
    der = json.loads((W1 / 'done' / f'{cell}_derive').read_text())
    s_ = out['sources']; cons = [l for l in range(len(s_)) if s_[l] != l]; order = der['revert_order']
    for lvl, child in (('S3', out), ('S2', isolate(out, sorted(set(cons) - set(order[:der['S2_reverted']])))),
                       ('S1', isolate(out, sorted(set(cons) - set(order[:der['S1_reverted']]))))):
        q = W2 / 'readers' / cell / lvl / f'{task}.pt'; q.parent.mkdir(parents=True, exist_ok=True)
        if q.exists(): q.unlink()
        torch.save(child, q)
    (W2 / 'done').mkdir(parents=True, exist_ok=True)
    for m in ('capture_fit', 'capture_val', 'train', 'derive'):
        shutil.copy(W1 / 'done' / f'{cell}_{m}', W2 / 'done' / f'{cell}_{m}')
    log.append(dict(cell=cell, consumers=len(cons), calibration=names, seconds=round(time.time() - t0, 1)))
    print(json.dumps(log[-1]), flush=True)
(W2 / 'mlr_build.json').write_text(json.dumps(dict(args=vars(a), cells=log), indent=1))
print('MLR_BUILD_DONE')
