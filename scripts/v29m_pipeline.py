"""多模型扩展（Qwen2.5-14B/32B）：每个 (task, budget) 格子的完整 RAPS 流程，沿用 7B v26/v29 的配方与冻结规则。
在模型专属 worktree 中运行（ROOT = 本文件上两级；runtime_data/assets 链接到该模型的前缀存储/pcache/索引）。
  capture : teacher 采集 fit/val（各 64 条，与 7B 相同的 data/v26/p0 查询；SUBJ 用 canonical），P8，identity
  plan    : 几何共享图（与 7B g16.json 同一构造：每个 P8 周期的 (p+1,p+2,p+3)/(p+5,p+6,p+7) 三层组、中间层为源，
            从最深层取，直到消费层 ≈ 3/7 L；L=28 时与 configs/v25_explore/g16.json 完全一致）
  train   : train_instrumented_v25.py，R4，200 步，eval-every 25，seed 24（与 7B 相同）
  derive  : frozen R0 导出；按 v29 规则的回退顺序（|l-s(l)| 降序，浅层优先）得到三档：S3（全部消费层，≈G16）、
            S2（回退 1/3，≈G20）、S1（回退 2/3，≈G24）
  dev     : 500 条开发集（data/v29/p0/dev），I/S1/S2/S3 端到端；按冻结 v29 规则选择（net ≥ -ceil(0.01n)，最小字节，0.5% 平局规则）
  test    : 完整测试集（studies/v28_frozen/bundles/<cell>/eval），I 与三档
用法: v29m_pipeline.py <stage|all> --cells trec_b25,... --gpu N [--cores a-b]
环境: PANS_MODEL_PATH, PANS_PY, PANS_PROBE_QUERY_HEADS, PANS_SELECTOR_KV_HEAD_IDS"""
from pathlib import Path
import argparse, json, math, os, subprocess, sys, time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
MODEL = os.environ['PANS_MODEL_PATH']; PY = os.environ['PANS_PY']
LEVELS = ('S1', 'S2', 'S3')
p = argparse.ArgumentParser(); p.add_argument('stage'); p.add_argument('--cells', required=True)
p.add_argument('--gpu', required=True); p.add_argument('--cores', default=None)
p.add_argument('--steps', type=int, default=200, help='reader 训练步数（7B 配方 200；大模型按验证集选最优步）')
p.add_argument('--tag', default='', help='结果目录 studies/v29m_<tag>')
p.add_argument('--reuse-from', default=None, help='复用该目录的 teacher 采集与 I 评估（与 reader 无关的步骤）')
a = p.parse_args()
W = ROOT / 'studies' / ('v29m' + (f'_{a.tag}' if a.tag else ''))
REUSE = (ROOT / 'studies' / a.reuse_from) if a.reuse_from else None

def reuse(name, rel):
    """把与 reader 无关的产物（teacher 记录、I 评估）从 REUSE 目录链接过来并标记完成。"""
    if REUSE is None or done(name) or not (REUSE / 'done' / name).exists(): return False
    src, dst = REUSE / rel, W / rel; dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists(): dst.symlink_to(src)
    mark(name, dict(reused_from=str(src))); print('REUSE', name, flush=True); return True

def chain(out, mode, bundle, samples, task, budget, adapters=None, diagnostic=False, admission='reuse-prefetch'):
    m = ['--mode', mode, '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{out}.extra.jsonl']
    if diagnostic: m.append('--diagnostic')
    if adapters: m += ['--adapters', str(adapters), '--allow-candidate', '--packed-ops']
    return ['--admission', admission, '--audit-jsonl', f'{out}.v26.jsonl', '--', '--model-path', MODEL,
            '--store-root', str(ROOT / 'runtime_data/prefix_store'), '--selector-index-dir', str(ROOT / 'assets/selector_index_k4_g32'),
            '--', '--pcache-root', str(ROOT / 'runtime_data/pcache_c16'), '--', *m, '--',
            '--task', task, '--budget', str(budget), '--period', '8', '--max-tokens', '1', '--warmup', '0', '--samples', str(samples),
            '--profile', 'none', '--bundle-dir', str(bundle), '--output', str(out)]

def sh(name, argv):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src')
    (W / 'logs').mkdir(parents=True, exist_ok=True); t0 = time.time()
    pre = ['taskset', '-c', a.cores] if a.cores else []
    with (W / 'logs' / f'{name}.log').open('x') as f:
        rc = subprocess.run(pre + argv, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    with (W / 'status.jsonl').open('a') as f: f.write(json.dumps(dict(name=name, returncode=rc, seconds=round(time.time() - t0, 1))) + '\n')
    print(name, rc, round(time.time() - t0), flush=True)
    if rc: raise SystemExit(f'{name} failed')

def done(name): return (W / 'done' / name).exists()
def stash(*paths):
    """上次中断留下、未标记完成的产物：挪到 _aborted/<时间戳>/ 留档，不删除。"""
    dst = W / '_aborted' / str(int(time.time()))
    for q in paths:
        q = Path(q)
        for x in sorted(q.parent.glob(q.name + '*')) if q.parent.exists() else []:
            dst.mkdir(parents=True, exist_ok=True); x.rename(dst / (x.parent.name + '__' + x.name)); print('STASH', x, flush=True)
def mark(name, obj=None):
    (W / 'done').mkdir(parents=True, exist_ok=True); (W / 'done' / name).write_text(json.dumps(obj or {}))

def geometry_map(L, records=None, period=8, frac=3 / 7):
    """三层组 (p+1,p+2,p+3)/(p+5,p+6,p+7)，中间层为源，从最深层取；消费层必须在全部 fit/val 记录中与源层选中位置完全相同
    （与 v26 规则一致的“几何相容”条件；预算取整可能让个别层的 chunk 数不同）。不相容的层跳过并继续往浅层取，
    直到消费层数达到 2*round(L*frac/2)。仅用选择几何，不看分数/标签。"""
    cands = []
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

def data_dir(task, kind):
    canon = task == 'subj'
    return ROOT / ('data/v26/p0/' + ('canonical_' if canon else '') + kind)

def stage_capture(cell):
    task, b = cell.split('_b')[0], int(cell.split('_b')[1])
    for split, role in (('fit', 'fit'), ('val', 'validation')):
        name = f'{cell}_capture_{split}'
        if done(name) or reuse(name, f'teacher/{cell}_{split}'): continue
        rec = W / 'teacher' / f'{cell}_{split}'
        out = W / 'capture' / f'{cell}_{split}'; out.parent.mkdir(parents=True, exist_ok=True)
        stash(rec, out, W / 'logs' / f'{name}.log')
        sh(name, [PY, '-u', 'scripts/collect_teacher_v26.py', '--teacher-records', str(rec), '--split', role, '--',
                  *chain(out, 'identity', data_dir(task, split), 64, task, b, diagnostic=True, admission='original')])
        mark(name)

def stage_plan_train(cell):
    task, b = cell.split('_b')[0], int(cell.split('_b')[1]); name = f'{cell}_train'
    if done(name): return
    from contiguous_fuxian.sharedkv_v24.records import load, FIELDS, PLAN_FORMAT, assert_plan
    fit, val = load(W / 'teacher' / f'{cell}_fit', task, 'fit'), load(W / 'teacher' / f'{cell}_val', task, 'validation')
    row = next(iter(fit.values())); L = row['geometry']['layers']; sources = geometry_map(L, list(fit.values()) + list(val.values()))
    if L == 28:
        assert sources == json.loads((ROOT / 'configs/v25_explore/g16.json').read_text())['sources'], 'geometry rule != 7B g16'
    for item in list(fit.values()) + list(val.values()): assert_plan(item, sources)
    plan = dict(format=PLAN_FORMAT, **{k: row[k] for k in FIELDS}, sources=sources, unique_sources=len(set(sources)),
                design_query_hashes=sorted(fit), status='v29m_geometry_only', quality_certified=False)
    pp = W / 'plans' / f'{cell}_plan.json'; pp.parent.mkdir(parents=True, exist_ok=True)
    stash(pp); pp.write_text(json.dumps(plan, indent=2))
    stash(W / 'ckpt' / cell, W / 'logs' / f'{name}.log')
    ck = W / 'ckpt' / cell / f'{task}.pt'; ck.parent.mkdir(parents=True, exist_ok=True)
    sh(name, [PY, '-u', 'scripts/train_instrumented_v25.py', '--native-blocks', '4', '--worklog', str(W / 'ckpt' / cell / 'compute.json'), '--',
              '--model-path', MODEL, '--store-root', str(ROOT / 'runtime_data/prefix_store'), '--task', task,
              '--teacher-fit', str(W / 'teacher' / f'{cell}_fit'), '--teacher-val', str(W / 'teacher' / f'{cell}_val'), '--plan', str(pp),
              '--steps', str(a.steps), '--eval-every', '25', '--seed', '24', '--device', 'cuda', '--output', str(ck)])
    mark(name, dict(layers=L, sources=sources, steps=a.steps))

def stage_derive(cell):
    task = cell.split('_b')[0]; name = f'{cell}_derive'
    if done(name): return
    import torch
    from contiguous_fuxian.sharedkv_v27.frozen_control import export_frozen_r0
    from contiguous_fuxian.sharedkv_v21.bundles import isolate
    stash(W / 'readers' / cell)
    r0 = W / 'readers' / cell / 'S3' / f'{task}.pt'; r0.parent.mkdir(parents=True, exist_ok=True)
    export_frozen_r0(W / 'ckpt' / cell / f'{task}.pt', r0)
    bnd = torch.load(r0, map_location='cpu', weights_only=False); s = bnd['sources']
    cons = [l for l in range(len(s)) if s[l] != l]; order = sorted(cons, key=lambda l: (-abs(l - s[l]), l))
    k1, k2 = round(len(cons) / 3), round(2 * len(cons) / 3)
    for lvl, k in (('S2', k1), ('S1', k2)):
        child = isolate(bnd, sorted(set(cons) - set(order[:k])))
        q = W / 'readers' / cell / lvl / f'{task}.pt'; q.parent.mkdir(parents=True, exist_ok=True)
        with q.open('xb') as f: torch.save(child, f)
    mark(name, dict(consumers=len(cons), revert_order=order, S2_reverted=k1, S1_reverted=k2))

def run_eval(cell, kind, arm):
    task, b = cell.split('_b')[0], int(cell.split('_b')[1]); name = f'{cell}_{kind}_{arm}'
    out = W / kind / name
    if arm == 'I': reuse(name, f'{kind}/{name}')
    if not done(name):
        bundle = ROOT / ('data/v29/p0/dev' if kind == 'dev' else f'studies/v28_frozen/bundles/{cell}/eval')
        n = sum(1 for _ in open(bundle / f'{task}.jsonl'))
        ad = None if arm == 'I' else W / 'readers' / cell / arm
        out.parent.mkdir(parents=True, exist_ok=True); stash(out, W / 'logs' / f'{name}.log')
        sh(name, [PY, 'scripts/run_evidence_v26.py', *chain(out, 'identity' if ad is None else 'shared', bundle, n, task, b, adapters=ad)])
        mark(name)
    recs = [json.loads(l) for l in open(out / 'scored_records.jsonl')]
    summ = json.loads((out / 'summary.json').read_text())
    return dict(n=len(recs), correct=sum(bool(r['correct']) for r in recs), preds={r['uid']: r['prediction'] for r in recs},
                bytes=list(summ['tasks'].values())[0].get('mean_physical_prefetch_kv_bytes'))  # 与 v29_a1_select.py 相同的读取位置

def stage_dev(cell):
    res = {arm: run_eval(cell, 'dev', arm) for arm in ('I',) + LEVELS}
    n = res['I']['n']; tol = math.ceil(0.01 * n); nconsumers = json.loads((W / 'done' / f'{cell}_derive').read_text())['consumers']
    cons = {'I': 0, 'S3': nconsumers}; d = json.loads((W / 'done' / f'{cell}_derive').read_text())
    cons['S2'] = nconsumers - d['S2_reverted']; cons['S1'] = nconsumers - d['S1_reverted']
    rows = {}
    for arm, r in res.items():
        net = r['correct'] - res['I']['correct']
        if r['bytes'] is None or not math.isfinite(r['bytes']): raise SystemExit(f'missing bytes {cell} {arm}')
        rows[arm] = dict(net=net, eligible=arm == 'I' or net >= -tol, bytes=r['bytes'], consumers=cons[arm],
                         disagree=sum(r['preds'][u] != res['I']['preds'][u] for u in r['preds']))
    el = {k: v for k, v in rows.items() if v['eligible']}; bmin = min(v['bytes'] for v in el.values())
    near = [k for k, v in el.items() if (v['bytes'] - bmin) / bmin < 0.005]
    order = {'I': 0, 'S1': 1, 'S2': 2, 'S3': 3}
    sel = sorted(near, key=lambda k: (-el[k]['net'], el[k]['consumers'], order[k]))[0]
    out = dict(cell=cell, n=n, tolerance=tol, selected=sel, candidates=rows)
    (W / 'selection').mkdir(parents=True, exist_ok=True); (W / 'selection' / f'{cell}.json').write_text(json.dumps(out, indent=1))
    print('SELECT', cell, sel, {k: (v['net'], round(v['bytes'] / 1e6, 1)) for k, v in rows.items()}, flush=True)

def stage_test(cell):
    res = {arm: run_eval(cell, 'test', arm) for arm in ('I',) + LEVELS}
    I = res['I']; out = {arm: dict(n=r['n'], acc=100 * r['correct'] / r['n'], delta_pp=100 * (r['correct'] - I['correct']) / r['n'],
                                   bytes=r['bytes'], disagree=100 * sum(r['preds'][u] != I['preds'][u] for u in r['preds']) / r['n'])
                         for arm, r in res.items()}
    (W / 'test_summary').mkdir(parents=True, exist_ok=True); (W / 'test_summary' / f'{cell}.json').write_text(json.dumps(out, indent=1))
    print('TEST', cell, {k: round(v['acc'], 2) for k, v in out.items()}, flush=True)

STAGES = dict(capture=stage_capture, train=stage_plan_train, derive=stage_derive, dev=stage_dev, test=stage_test)
for cell in a.cells.split(','):
    for st in (STAGES if a.stage == 'all' else [a.stage]):
        STAGES[st](cell)
print('PIPELINE_DONE', a.stage, a.cells, flush=True)
