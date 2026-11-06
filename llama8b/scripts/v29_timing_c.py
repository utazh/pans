"""v29 stage C timing: arms I (our no-sharing system, bridge to stage B), CKV (ContiguousKV reproduction),
IMP (IMPRESS reproduction). Same protocol as stage B / rule rev 3: GPU2, cores 28-31,60-63, SSD Pcache copies,
preserve (SSD-cold, 16 cells) + warm (25%/50% cells), 6 blocks x 16 queries, all 6 orders of 3 arms,
gate before each block, attributable-only flags, flagged/failed block rerun once. Usage: v29_timing_c.py [--dry-run]"""
from pathlib import Path
import itertools, json, os, subprocess, sys, threading, time, hashlib
R = Path('/home/shensicheng/qyd/pans/src'); PY = '/home/shensicheng/qyd/pans/venv/bin/python'
GPU, CORES = '2', '28-31,60-63'
TIMING = set(range(28, 32)) | set(range(60, 64)); BACKGROUND = set(range(0, 12)) | set(range(16, 28))
NUMA1 = set(range(16, 32)) | set(range(48, 64))
C16 = R / 'runtime_data/v29_pcache_gpu2_ssd'; C64 = R / 'runtime_data/v29_pcache_impress_c64_ssd'
REORDER = R / 'configs/v29_impress_reorder_p0_fit4.json'
DRY = '--dry-run' in sys.argv
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze = json.loads((R / 'audit/v29_rule_freeze.json').read_text())
assert sha(R / 'configs/v29_selection_rule.json') == freeze['sha256'] and freeze['revision'] == 3
OUT = R / ('studies/v29/timing_c_dryrun' if DRY else 'studies/v29/timing_c')
OUT.mkdir(parents=True, exist_ok=False); (OUT / 'runs').mkdir(); (OUT / 'logs').mkdir()
CELLS = [f'{t}_b{b}' for t in ('sst2', 'subj', 'trec', 'rte') for b in (5, 10, 25, 50)]

def argv(cell, arm, cond, block, name):
    task, b = cell.split('_')[0], int(cell.split('_b')[1]); o = OUT / 'runs' / name
    kv = C64 if arm == 'IMP' else C16
    cold = ['--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(kv), '--cold-log', f'{o}.cold.jsonl'] if cond == 'preserve' else []
    bundle = str(R / f'studies/v28_frozen/bundles/{cell}/b{block:02d}')
    tail = ['--task', task, '--budget', str(b), '--max-tokens', '1', '--warmup', '4', '--samples', '16', '--bundle-dir', bundle, '--output', str(o)]
    if arm == 'I':
        return [PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
                '--model-path', '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'),
                '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(kv), *cold, '--',
                '--mode', 'identity', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl', '--',
                '--period', '8', '--profile', 'none', *tail]
    method = ['--method', 'contigkv'] if arm == 'CKV' else ['--method', 'impress', '--impress-reorder', str(REORDER)]
    return [PY, str(R / 'scripts/v29_baseline_entry.py'), '--admission', 'original', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'), '--',
            '--pcache-root', str(kv), *cold, '--', '--mode', 'original', '--', *method, *tail]

plan = []
orders = list(itertools.permutations(['I', 'CKV', 'IMP']))
for cond in ('preserve', 'warm'):
    for cell in CELLS:
        if cond == 'warm' and int(cell.split('_b')[1]) not in (25, 50): continue
        for block in range(6): plan.append(dict(cell=cell, cond=cond, block=block, order=list(orders[block])))
(OUT / 'plan.json').write_text(json.dumps(dict(rule_sha256=freeze['sha256'], c16=str(C16), c64=str(C64), reorder_sha256=sha(REORDER),
    gpu=GPU, cores=CORES, blocks=plan, runs=3 * len(plan)), indent=1))
print('planned blocks', len(plan), 'runs', 3 * len(plan), flush=True)
if DRY:
    for b in plan[:2]: print(b)
    print(' '.join(argv('sst2_b5', 'IMP', 'preserve', 0, 'x'))); sys.exit(0)

HZ = os.sysconf('SC_CLK_TCK')
def proc_table():
    out = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            st = (p / 'stat').read_text(); rest = st[st.rindex(')') + 2:].split()
            out[int(p.name)] = dict(ppid=int(rest[1]), cpu=(int(rest[11]) + int(rest[12])) / HZ, last_cpu=int(rest[36]))
        except Exception: pass
    return out
def descendants(root, table):
    kids = {}; [kids.setdefault(v['ppid'], []).append(k) for k, v in table.items()]
    seen, stack = set(), [root]
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x); stack += kids.get(x, [])
    return seen
def gpu_procs():
    try:
        uu = dict(l.split(', ') for l in subprocess.run(['nvidia-smi', '--query-gpu=gpu_uuid,index', '--format=csv,noheader'],
                   capture_output=True, text=True, timeout=20).stdout.strip().splitlines())
        rows = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return [(uu.get(u), int(p)) for u, p in (r.split(', ') for r in rows if r)]
    except Exception as e: return [('error', repr(e))]
def totals():
    cpu = {}
    for l in Path('/proc/stat').read_text().splitlines():
        if l.startswith('cpu') and l[3:4].isdigit():
            f = l.split(); c = int(f[0][3:]); v = list(map(int, f[1:])); cpu[c] = (sum(v) - v[3] - v[4], sum(v))
    sda = [l.split() for l in Path('/proc/diskstats').read_text().splitlines() if l.split()[2] == 'sda'][0]
    return dict(numa1=cpu, sda_bytes=(int(sda[5]) + int(sda[9])) * 512)
class Telemetry(threading.Thread):
    def __init__(self, log):
        super().__init__(daemon=True); self.log, self.stop, self.root = log, threading.Event(), None
    def run(self):
        prev = proc_table(); prev_t = time.time()
        with self.log.open('a') as f:
            while not self.stop.wait(5):
                now = proc_table(); t = time.time(); dt = t - prev_t
                ours = (descendants(self.root, now) if self.root else set()) | descendants(os.getpid(), now)
                foreign = sum(max(0, v['cpu'] - prev[k]['cpu']) for k, v in now.items() if k in prev and k not in ours and v['last_cpu'] in TIMING) / dt
                fg = [(g, p) for g, p in gpu_procs() if g in ('2', '3') and p not in ours]
                f.write(json.dumps(dict(t=t, foreign_timing_cores=foreign, foreign_gpu23=fg, totals=totals())) + '\n'); f.flush()
                prev, prev_t = now, t
def gate(window=30):
    a = totals(); time.sleep(window); b = totals()
    def cores(cs): return sum((b['numa1'][c][0] - a['numa1'][c][0]) / max(1, b['numa1'][c][1] - a['numa1'][c][1]) for c in cs)
    sda = (b['sda_bytes'] - a['sda_bytes']) / window / 1e6; gp = [(g, p) for g, p in gpu_procs() if g in ('2', '3')]
    bt = cores(TIMING)
    return dict(timing_cores_busy=bt, background_cpus_busy=cores(BACKGROUND), numa1_total_busy_cores=cores(NUMA1),
                sda_total_MBps=sda, gpu23_procs=gp, passed=bt < 1 and sda < 5 and not gp)
def flagged(tlog, t0, t1):
    rows = [json.loads(l) for l in tlog.read_text().splitlines()] if tlog.exists() else []
    rows = [r for r in rows if t0 <= r['t'] <= t1]
    if any(r['foreign_gpu23'] for r in rows): return 'foreign_gpu23_process'
    for i in range(len(rows)):
        win = [r['foreign_timing_cores'] for r in rows if rows[i]['t'] <= r['t'] < rows[i]['t'] + 30]
        if len(win) >= 5 and sum(win) / len(win) >= 1: return 'foreign_timing_cores>=1core_30s'
    return None
env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', PYTHONPATH='src:runtime_deps_v24')
tel = Telemetry(OUT / 'telemetry.jsonl'); tel.start()
def run_block(b, suffix):
    waits = 0
    while True:
        g = gate(); g['t'] = time.time()
        with (OUT / 'gates.jsonl').open('a') as f: f.write(json.dumps(dict(g, cell=b['cell'], cond=b['cond'], block=b['block'], suffix=suffix)) + '\n')
        if g['passed']: break
        waits += 1; time.sleep(60)
    t0 = time.time(); results = []
    for pos, arm in enumerate(b['order']):
        name = f"{b['cell']}_{b['cond']}_{arm}_b{b['block']:02d}{suffix}"; s0 = time.time()
        with (OUT / 'logs' / f'{name}.log').open('x') as f:
            p = subprocess.Popen(['taskset', '-c', CORES] + argv(b['cell'], arm, b['cond'], b['block'], name), cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT)
            tel.root = p.pid; rc = p.wait(); tel.root = None
        results.append(dict(name=name, arm=arm, position=pos, returncode=rc, seconds=round(time.time() - s0, 1)))
    t1 = time.time(); flag = flagged(OUT / 'telemetry.jsonl', t0, t1)
    rec = dict(cell=b['cell'], cond=b['cond'], block=b['block'], order=b['order'], suffix=suffix, gate_waits=waits,
               t0=t0, t1=t1, flag=flag, runs=results, all_ok=all(r['returncode'] == 0 for r in results))
    with (OUT / 'blocks.jsonl').open('a') as f: f.write(json.dumps(rec) + '\n')
    print(json.dumps({k: rec[k] for k in ('cell', 'cond', 'block', 'suffix', 'flag', 'all_ok')}), flush=True)
    return rec
for b in plan:
    rec = run_block(b, '')
    if rec['flag'] or not rec['all_ok']: run_block(b, '_r1')
tel.stop.set()
(OUT / 'done.json').write_text(json.dumps(dict(finished=time.time()))); print('TIMING_C_DONE', flush=True)
