"""P2 并发/吞吐实验（7B，25% 预算，SSD 冷）。
N 个 worker 同时运行未修改的计时 runner（与 v29m_timing_sys 相同的命令行与冷认证），每个 worker：
  - 读自己的一份 pcache 副本（真正互不命中的冷读，模拟 N 个不同前缀同时恢复）；
  - 服务同一任务的 48 条不同问题（96 条计时问题按 worker 轮转），闭环、背靠背；
  - 绑定 3 个独立物理核；GPU 按 [0,1,2,4] 轮转（N=8 时每卡 2 个）。
驱动每 0.25 s 轮询各 worker 的 requests.progress.jsonl 行数与 sda 读字节，给每个请求打完成时间；
报告只用“所有 worker 都处于请求阶段”的稳态窗口：吞吐 = 窗口内完成请求数 / 窗口长度，TTFT 取窗口内完成的请求。
用法（cwd=src，已 source env_timing.sh）: conc.py <study> --cells trec_b25,sst2_b25 --arms I,SEL,CKV --ns 1,2,4,8 --reps 2 [--gpus 0,1,2,4] [--smoke]"""
from pathlib import Path
import json, os, random, shutil, subprocess, sys, time
R = Path.cwd(); B = Path('/home/shensicheng/qyd/pans'); PY, MODEL = os.environ['PANS_PY'], os.environ['PANS_MODEL_PATH']
args = sys.argv[1:]; study = args[0]; opt = lambda k, d=None: args[args.index(k) + 1] if k in args else d
CELLS = opt('--cells', 'trec_b25,sst2_b25').split(','); ARMS = opt('--arms', 'I,SEL,CKV').split(',')
NS = [int(x) for x in opt('--ns', '1,2,4,8').split(',')]; REPS = int(opt('--reps', '2')); GPUS = opt('--gpus', '0,1,2,4').split(',')
SMOKE = '--smoke' in args; NQ = 8 if SMOKE else 48
C16 = (R / 'runtime_data/v29_pcache_gpu2_ssd').resolve(); C64 = B / 'work/q7/pcache_impress_c64_timing'; REORDER = B / 'work/q7/impress_reorder_fit4.json'
CW = B / 'work/q7/conc'; OUT = R / 'studies/v29' / ('conc_' + study)   # runner 要求输出在 ROOT 下
final = json.loads((R / 'audit/v29_final_selection.json').read_text())['cells']
fix = lambda p: str(p).replace('/home/panzihang/src/prism_layerkv', str(R))
def sel_arm(cell):
    assert final[cell]['config'] != 'I', cell                     # 并发实验只用选中了共享的单元
    return final[cell]['config'], fix(final[cell]['adapter_dir'])
cores = lambda w: f'{100 + 3 * w}-{102 + 3 * w}'                  # 100..123 物理核，避开他人绑核与 SMT 兄弟

def pcache_copy(kind, w):
    src = C64 if kind == 'c64' else C16; dst = CW / f'pc_{kind}_w{w}'
    if not (dst / '.contiguous_fuxian_complete').exists():
        if dst.exists(): shutil.rmtree(dst)
        subprocess.run(['cp', '-a', str(src), str(dst)], check=True)
    return dst

def bundle(cell, rep, w):
    task = cell.split('_')[0]; d = R / 'studies/v29/conc_bundles' / f'{cell}_q{NQ}_r{rep}_w{w}'
    if not (d / f'{task}.jsonl').exists():
        rows = [l for b in range(6) for l in (R / f'studies/v28_frozen/bundles/{cell}/b{b:02d}/{task}.jsonl').read_text().splitlines() if l.strip()]
        off = (12 * w + 24 * rep) % len(rows); pick = (rows[off:] + rows[:off])[:NQ]
        d.mkdir(parents=True, exist_ok=True); (d / f'{task}.jsonl').write_text(''.join(l + '\n' for l in pick))
    return d

def argv(cell, arm, w, rep, o):
    task, b = cell.split('_')[0], int(cell.split('_b')[1]); kv = pcache_copy('c64' if arm == 'IMP' else 'c16', w)
    cold = ['--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(kv), '--cold-log', f'{o}.cold.jsonl']
    tail = ['--task', task, '--budget', str(b), '--max-tokens', '1', '--warmup', '4', '--samples', str(NQ),
            '--bundle-dir', str(bundle(cell, rep, w)), '--output', str(o)]
    if arm in ('CKV', 'IMP'):
        method = ['--method', 'contigkv'] if arm == 'CKV' else ['--method', 'impress', '--impress-reorder', str(REORDER)]
        return [PY, str(R / 'scripts/v29_baseline_entry.py'), '--admission', 'original', '--audit-jsonl', f'{o}.v26.jsonl', '--',
                '--model-path', MODEL, '--store-root', str(R / 'runtime_data/prefix_store'), '--',
                '--pcache-root', str(kv), *cold, '--', '--mode', 'original', '--', *method, *tail]
    mode = ['--mode', 'identity' if arm == 'I' else 'shared', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl']
    if arm == 'SEL': mode += ['--adapters', sel_arm(cell)[1], '--allow-candidate', '--packed-ops']
    return [PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', MODEL, '--store-root', str(R / 'runtime_data/prefix_store'),
            '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(kv), *cold, '--', *mode, '--',
            '--period', '8', '--profile', 'none', *tail]

def sda_read():
    f = [l.split() for l in Path('/proc/diskstats').read_text().splitlines() if l.split()[2] == 'sda'][0]
    return int(f[5]) * 512, int(f[9]) * 512
def gpu_foreign(gpus):
    uu = dict(l.split(', ') for l in subprocess.run(['nvidia-smi', '--query-gpu=gpu_uuid,index', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().splitlines())
    rows = [r.split(', ') for r in subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().splitlines() if r]
    return [(uu.get(u), p) for u, p in rows if uu.get(u) in gpus]
def quiet(gpus, window=20):
    while True:
        a = sda_read(); time.sleep(window); b = sda_read()
        mbps = (sum(b) - sum(a)) / window / 1e6; fg = gpu_foreign(gpus)
        if mbps < 5 and not fg: return mbps
        print(f'  waiting: sda {mbps:.1f} MB/s, foreign gpu procs {fg}', flush=True); time.sleep(40)

def lines(p):
    try: return p.read_bytes().count(b'\n')
    except FileNotFoundError: return 0

def run_config(cell, arm, n, rep):
    name = f'{cell}_{arm}_n{n}_r{rep}'
    if (OUT / 'configs' / f'{name}.json').exists(): return
    gpus = [GPUS[w % len(GPUS)] for w in range(n)]; pre = None if SMOKE else quiet(sorted(set(gpus)))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='3', PYTHONPATH='src:runtime_deps_v24')
    procs, outs = [], []
    for w in range(n):
        o = OUT / 'runs' / f'{name}_w{w}'; outs.append(o)
        log = (OUT / 'logs' / f'{name}_w{w}.log').open('x')
        procs.append(subprocess.Popen(['taskset', '-c', cores(w)] + argv(cell, arm, w, rep, o), cwd=R,
                                      env=dict(env, CUDA_VISIBLE_DEVICES=gpus[w]), stdout=log, stderr=subprocess.STDOUT))
    ev = []; seen = [0] * n; t_start = time.time()
    while any(p.poll() is None for p in procs):
        t = time.time(); rd, wr = sda_read()
        for w, o in enumerate(outs):
            k = lines(o / 'requests.progress.jsonl')
            if k > seen[w]: ev.append(dict(t=t, w=w, done=k)); seen[w] = k
        ev.append(dict(t=t, sda_read=rd, sda_write=wr))
        time.sleep(0.25)
    rcs = [p.returncode for p in procs]
    rec = dict(cell=cell, arm=arm, n=n, rep=rep, gpus=gpus, cores=[cores(w) for w in range(n)], pre_sda_MBps=pre,
               t_start=t_start, t_end=time.time(), returncodes=rcs, events=ev)
    (OUT / 'configs' / f'{name}.json').write_text(json.dumps(rec))
    print(json.dumps(dict(name=name, rc=rcs, secs=round(rec['t_end'] - t_start), done=seen)), flush=True)

if __name__ == '__main__':
    for d in ('runs', 'logs', 'configs'): (OUT / d).mkdir(parents=True, exist_ok=True)
    plan = [(c, a, n, r) for r in range(REPS) for n in NS for c in CELLS for a in ARMS]
    random.Random(2026).shuffle(plan)                              # 打乱顺序，避免系统与时间漂移混淆
    (OUT / 'plan.json').write_text(json.dumps(dict(cells=CELLS, arms=ARMS, ns=NS, reps=REPS, gpus=GPUS, nq=NQ, plan=plan,
                                                   sel={c: sel_arm(c) for c in CELLS}, model=MODEL), indent=1))
    for w in range(max(NS)):
        pcache_copy('c16', w)
        if 'IMP' in ARMS: pcache_copy('c64', w)
    for cfg in plan: run_config(*cfg)
    (OUT / 'done.json').write_text(json.dumps(dict(finished=time.time()))); print('CONC_DONE', flush=True)
