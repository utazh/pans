"""G3 模型级读取器（留一法）：用若干“训练任务”的前缀与校准问题一起训练同一套读取器，导出给一个“目标任务”使用；
目标任务的前缀、校准问题、开发/测试数据都不参与训练与 checkpoint 选择（只取它的共享图与身份字段）。
完全复用 v24 训练模块（经 v25 training_patch：R4 补丁、相同损失、相同导出格式），不修改任何受 guard 保护的文件：
  初始化 = 所有训练任务前缀拼接后的闭式带正则最小二乘；训练 = 在各训练任务的行上随机采样；最优步 = 训练任务验证集平均目标值。
用法（cwd = 模型 worktree）:
  train_multi_v24.py --w studies/v29m_s800 --train-cells sst2_b25,trec_b25,rte_b25 --target subj_b25 --out-w studies/v29m_mlt --steps 800"""
import argparse, copy, json, random, sys, time
from pathlib import Path
import torch
from torch import nn
R = Path.cwd(); sys.path.insert(0, str(R / 'src'))
from contiguous_fuxian.sharedkv_v25.guard import check
from contiguous_fuxian.sharedkv_v25.train import training_patch
from contiguous_fuxian.sharedkv_v25.policy import NativePolicy

p = argparse.ArgumentParser()
p.add_argument('--model-path', required=True); p.add_argument('--store-root', default='runtime_data/prefix_store')
p.add_argument('--w', required=True); p.add_argument('--train-cells', required=True); p.add_argument('--target', required=True)
p.add_argument('--out-w', required=True); p.add_argument('--steps', type=int, default=800); p.add_argument('--eval-every', type=int, default=50)
p.add_argument('--native-blocks', type=int, default=4); p.add_argument('--seed', type=int, default=24); p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--ridge', type=float, default=1e-3); p.add_argument('--trust-weight', type=float, default=1e-3)
p.add_argument('--max-sequence', type=int, default=256); p.add_argument('--temperature', type=float, default=1.)
p.add_argument('--score-weight', type=float, default=1.); p.add_argument('--token-weight', type=float, default=.1)
p.add_argument('--rank-weight', type=float, default=.25); p.add_argument('--margin-cap', type=float, default=.5)
a = p.parse_args()
check(R)
W, OUT = R / a.w, R / a.out_w
tcell = a.target; ttask = tcell.split('_b')[0]; train_cells = a.train_cells.split(',')
assert tcell not in train_cells, 'target must be held out'
started = time.perf_counter()

with training_patch(NativePolicy(a.native_blocks)) as t:
    sets = {}
    for cell in train_cells:
        task = cell.split('_b')[0]
        fit, val = t.load(W / 'teacher' / f'{cell}_fit', task, 'fit'), t.load(W / 'teacher' / f'{cell}_val', task, 'validation')
        t.assert_disjoint(fit, val)
        plan = json.loads((W / 'plans' / f'{cell}_plan.json').read_text()); src = plan['sources']
        t.validate_sources(src, plan['period'])
        for row in list(fit.values()) + list(val.values()): t.assert_plan(row, src)
        sets[task] = dict(cell=cell, fit=fit, val=val, sources=src, header=next(iter(fit.values())))
    tplan = json.loads((W / 'plans' / f'{tcell}_plan.json').read_text())
    theader = next(iter(t.load(W / 'teacher' / f'{tcell}_fit', ttask, 'fit').values()))   # 只取身份字段，不用其问题
    # 读取器按 (消费层, 源层) 对索引：不同任务的共享图可以不同（选块相容性随前缀而变），同一层对共用同一个读取器
    pair = lambda l, s: f'{l}_{s}'
    pairs = sorted({(l, s) for src in [s_['sources'] for s_ in sets.values()] + [tplan['sources']] for l, s in enumerate(src) if l != s})
    trained_pairs = {(l, s) for s_ in sets.values() for l, s in enumerate(s_['sources']) if l != s}
    tcons = [l for l, s in enumerate(tplan['sources']) if l != s]
    untrained = [(l, tplan['sources'][l]) for l in tcons if (l, tplan['sources'][l]) not in trained_pairs]
    print(json.dumps(dict(pairs=len(pairs), target_consumers=len(tcons), target_pairs_without_training_rows=untrained)), flush=True)
    def view(src):                               # 某个任务共享图下的 {消费层: 读取器}
        return {str(l): readers[pair(l, s)] for l, s in enumerate(src) if l != s}

    from transformers import AutoModelForCausalLM
    device = torch.device('cuda'); dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model_path, local_files_only=True, dtype=dtype, attn_implementation='eager').to(device).eval()
    model.requires_grad_(False)
    L = len(tplan['sources'])
    raw_cpu = {}
    for task, s in sets.items():
        raw_cpu[task] = {}
        for lid in range(L):
            vals = []
            for kind in ('key', 'value'):
                x, meta = t.read_prefix(a.store_root, task, lid, kind)
                if meta['token_hash'] != s['header']['token_hash']: raise ValueError('Prefix identity mismatch')
                vals.append(x)
            raw_cpu[task][lid] = tuple(vals)
    heads = theader['geometry']['kv_heads']; readers = {}
    for lid, src in pairs:                       # 闭式初始化：训练任务前缀拼接（目标前缀不参与）
        M = {}
        for ki, (wm, wb) in enumerate((('A', 'bk'), ('B', 'bv'))):
            xs = torch.cat([raw_cpu[k][src][ki].float() for k in sets]); xl = torch.cat([raw_cpu[k][lid][ki].float() for k in sets])
            mm, bb = zip(*(t.affine_ridge(xs[:, h], xl[:, h], ridge=a.ridge) for h in range(heads)))
            M[wm] = torch.stack(mm); M[wb] = torch.stack(bb)
        readers[pair(lid, src)] = t.TrainableReader(t.Reader(M['A'], M['B'], M['bk'], M['bv']), ste=True)
    readers = nn.ModuleDict(readers).to(device)
    raw = {k: {l: tuple(x.to(device=device, dtype=dtype) for x in kv) for l, kv in v.items()} for k, v in raw_cpu.items()}; del raw_cpu

    def evaluate_all():
        tot, n, flips, per = 0., 0, 0, {}
        for task, s in sets.items():
            r = t.evaluate(model, s['val'], raw[task], s['sources'], view(s['sources']), a)
            tot += r['objective'] * r['requests']; n += r['requests']; flips += r['teacher_decision_flips']
            per[task] = dict(objective=r['objective'], flips=r['teacher_decision_flips'])
        return dict(objective=tot / n, teacher_decision_flips=flips, requests=n, per_task=per)

    init_params = [x.detach().clone() for x in readers.parameters()]
    rep = evaluate_all(); print(json.dumps({'step': 0, **rep}), flush=True)
    best = copy.deepcopy(readers.state_dict()); best_rep = rep; best_step = 0; history = [{'step': 0, **rep}]
    opt = torch.optim.Adam(readers.parameters(), lr=a.lr); rng = random.Random(a.seed); torch.manual_seed(a.seed)
    rows = [(task, row) for task, s in sets.items() for row in s['fit'].values()]
    for step in range(1, a.steps + 1):
        task, row = rows[rng.randrange(len(rows))]; s = sets[task]
        bank = t.selected_bank(raw[task], row, s['sources'], device, dtype)
        labels = row['label_order']; target = t.teacher_vectors(row, device)
        rv = view(s['sources'])
        def compute(label, row=row, s=s, bank=bank, rv=rv):
            return t.continuation_tokens(model, row, label, s['sources'], bank, rv, checkpoint_layers=True, max_sequence=a.max_sequence)
        opt.zero_grad(set_to_none=True)
        t.streamed_backward(compute, labels, target, t.settings(a))
        norm = sum(x.square().sum() for x in init_params).clamp_min(1.)
        trust = sum((q - q0).square().sum() for q, q0 in zip(readers.parameters(), init_params)) / norm
        if a.trust_weight: (a.trust_weight * trust).backward()
        if any(x.grad is not None for x in model.parameters()): raise RuntimeError('Backbone received gradients')
        torch.nn.utils.clip_grad_norm_(readers.parameters(), 1.); opt.step(); del bank
        if step % a.eval_every == 0 or step == a.steps:
            rep = evaluate_all(); history.append({'step': step, **rep}); print(json.dumps(history[-1]), flush=True)
            if rep['objective'] < best_rep['objective']: best = copy.deepcopy(readers.state_dict()); best_rep = rep; best_step = step
    readers.load_state_dict(best)
    with torch.no_grad():
        for q in readers.parameters(): q.copy_(q.to(torch.bfloat16).float())
    hashes_fit = sorted(h for s in sets.values() for h in s['fit']); hashes_val = sorted(h for s in sets.values() for h in s['val'])
    bundle = {'format': t.BUNDLE_FORMAT, 'base_commit': t.BASE_COMMIT, **{k: theader[k] for k in t.FIELDS},
              'sources': tplan['sources'], 'readers': {str(l): readers[pair(l, tplan['sources'][l])].export().state_dict() for l in tcons},
              'target_pairs_without_training_rows': untrained,
              'calibration_query_hashes': sorted(set(hashes_fit) | set(hashes_val)), 'fit_query_hashes': hashes_fit,
              'validation_query_hashes': hashes_val, 'diagnostic_only': True, 'deployment_status': 'score_candidate_not_accepted',
              'accepted_consumers': 0, 'candidate_consumers': len(tcons), 'end_to_end_validated': False, 'local_gate_passed': False,
              'compiler_family': 'v24_score_aligned_multi_prefix_heldout', 'best_step': best_step, 'report': best_rep,
              'compiler_args': vars(a), 'topology_digest': t.json_digest(tplan), 'parent_report': None,
              'model_level': dict(train_cells=train_cells, held_out=tcell, note='target prefix/queries unused in fit, init, and checkpoint choice'),
              'scope': 'Model-level readers: multi-prefix training on other tasks; held-out target'}
    out = OUT / 'ckpt' / tcell / f'{ttask}.pt'; out.parent.mkdir(parents=True, exist_ok=True)
    t.save_new(out, bundle)
    secs = time.perf_counter() - started
    (OUT / 'ckpt' / tcell / 'compute.json').write_text(json.dumps(dict(status='completed', wall_seconds_total_including_load=secs,
        best_step=best_step, history=history, train_cells=train_cells, held_out=tcell), indent=1))
    print(json.dumps({'output': str(out), 'best_step': best_step, 'seconds': round(secs)}), flush=True)
