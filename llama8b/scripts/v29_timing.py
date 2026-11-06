"""v29 stage B timing (rule rev 2, final selection frozen). GPU2 exclusive, serial, cores 16-31,48-63,
SSD pcache copy. Arms per cell: I, G16 (frozen v28 R0), selected (if not I/G16).
Conditions: preserve (SSD-cold, all 16 cells) and warm (ordinary cache, 25%/50% cells), 6 blocks x 16 queries.
3 arms: the 6 blocks use all 6 orders; 2 arms: AB/BA alternate.
Telemetry every 5 s; pre-start gate before each block; blocks flagged on attributable evidence only are
rerun once (whole block). Never judged by TTFT. Usage: v29_timing.py [--dry-run] [--cells a,b] [--conditions preserve,warm]"""
from pathlib import Path
import itertools, json, os, subprocess, sys, threading, time, hashlib
R = Path('/home/shensicheng/qyd/pans/src'); PY = '/home/shensicheng/qyd/pans/venv/bin/python'
GPU, CORES = '2', '28-31,60-63'  # rule rev 3: NUMA1 cores free of the background threads
TIMING = set(range(28, 32)) | set(range(60, 64))
BACKGROUND = set(range(0, 12)) | set(range(16, 28))
NUMA1 = set(range(16, 32)) | set(range(48, 64))
PCACHE = R / 'runtime_data/v29_pcache_gpu2_ssd'
args = sys.argv[1:]
DRY = '--dry-run' in args
opt = lambda k: args[args.index(k) + 1].split(',') if k in args else None
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
final = json.loads((R / 'audit/v29_final_selection.json').read_text())
man = json.loads((R / 'studies/v29/readers/manifest.json').read_text())['cells']
freeze = json.loads((R / 'audit/v29_rule_freeze.json').read_text())
assert sha(R / 'configs/v29_selection_rule.json') == freeze['sha256'] and freeze['revision'] == 3
OUT = R / ('studies/v29/timing_dryrun' if DRY else 'studies/v29/timing')
OUT.mkdir(parents=True, exist_ok=False); (OUT / 'runs').mkdir(); (OUT / 'logs').mkdir()

def arms_of(cell):
    sel = final['cells'][cell]['config']; arms = {'I': None, 'G16': man[cell]['G16']['adapter_dir']}
    if sel not in arms: arms[sel] = final['cells'][cell]['adapter_dir']
    return arms

def argv(cell, arm, adapter, cond, block, name):
    task, b = cell.split('_')[0], int(cell.split('_b')[1]); o = OUT / 'runs' / name
    cold = ['--cold', 'ssd-cold', '--read-advice', 'preserve', '--owned-cache-root', str(PCACHE), '--cold-log', f'{o}.cold.jsonl'] if cond == 'preserve' else []
    mode = ['--mode', 'identity' if arm == 'I' else 'shared', '--backend', 'vllm2', '--metadata-cache', '--layout', 'hybrid', '--stats', f'{o}.extra.jsonl']
    if arm != 'I': mode += ['--adapters', adapter, '--allow-candidate', '--packed-ops']
    return [PY, str(R / 'scripts/run_evidence_v26.py'), '--admission', 'reuse-prefetch', '--audit-jsonl', f'{o}.v26.jsonl', '--',
            '--model-path', '/home/shensicheng/qyd/pans/models/Qwen2.5-7B-Instruct', '--store-root', str(R / 'runtime_data/prefix_store'),
            '--selector-index-dir', str(R / 'assets/selector_index_k4_g32'), '--', '--pcache-root', str(PCACHE), *cold, '--', *mode, '--',
            '--task', task, '--budget', str(b), '--period', '8', '--max-tokens', '1', '--warmup', '4', '--samples', '16',
            '--profile', 'none', '--bundle-dir', str(R / f'studies/v28_frozen/bundles/{cell}/b{block:02d}'), '--output', str(o)]

cells = opt('--cells') or list(final['cells'])
conds = opt('--conditions') or ['preserve', 'warm']
plan = []
for cond in conds:
    for cell in cells:
        if cond == 'warm' and int(cell.split('_b')[1]) not in (25, 50): continue
        arms = arms_of(cell); names = list(arms)
        orders = list(itertools.permutations(names)) if len(names) == 3 else [tuple(names), tuple(reversed(names))] * 3
        for block in range(6):
            plan.append(dict(cell=cell, cond=cond, block=block, order=list(orders[block]), arms=arms))
(OUT / 'plan.json').write_text(json.dumps(dict(final_selection_sha256=sha(R / 'audit/v29_final_selection.json'),
    rule_sha256=freeze['sha256'], pcache=str(PCACHE), gpu=GPU, cores=CORES, blocks=plan,
    runs=sum(len(b['order']) for b in plan)), indent=1))
print('planned blocks', len(plan), 'runs', sum(len(b['order']) for b in plan), flush=True)
if DRY:
    for b in plan[:3] + plan[-2:]: print(b['cell'], b['cond'], b['block'], b['order'])
    sys.exit(0)

# ---------------- telemetry ----------------
HZ = os.sysconf('SC_CLK_TCK'); ME = os.getuid()
def proc_table():
    out = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            st = (p / 'stat').read_text(); rest = st[st.rindex(')') + 2:].split()
            out[int(p.name)] = dict(ppid=int(rest[1]), cpu=(int(rest[11]) + int(rest[12])) / HZ, last_cpu=int(rest[36]),
                                    uid=(p / 'status').stat().st_uid)
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
                ours = descendants(self.root, now) if self.root else set()
                ours |= descendants(os.getpid(), now)
                foreign_timing = sum(max(0, v['cpu'] - prev[k]['cpu']) for k, v in now.items()
                                     if k in prev and k not in ours and v['last_cpu'] in TIMING) / dt
                foreign_gpu = [(g, p) for g, p in gpu_procs() if g in ('2', '3') and p not in ours]
                f.write(json.dumps(dict(t=t, foreign_timing_cores=foreign_timing, foreign_gpu23=foreign_gpu, totals=totals())) + '\n'); f.flush()
                prev, prev_t = now, t

def gate(window=30):
    a = totals(); time.sleep(window); b = totals()
    def cores(cs):
        d = [(b['numa1'][c][0] - a['numa1'][c][0], b['numa1'][c][1] - a['numa1'][c][1]) for c in cs]
        return sum(x / max(1, y) for x, y in d)
    busy_t, busy_bg, busy_n1 = cores(TIMING), cores(BACKGROUND), cores(NUMA1)
    sda = (b['sda_bytes'] - a['sda_bytes']) / window / 1e6
    gp = [(g, p) for g, p in gpu_procs() if g in ('2', '3')]
    return dict(timing_cores_busy=busy_t, background_cpus_busy=busy_bg, numa1_total_busy_cores=busy_n1,
                sda_total_MBps=sda, gpu23_procs=gp, passed=busy_t < 1 and sda < 5 and not gp)

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
        name = f"{b['cell']}_{b['cond']}_{arm}_b{b['block']:02d}{suffix}"
        a = argv(b['cell'], arm, b['arms'][arm], b['cond'], b['block'], name); s0 = time.time()
        with (OUT / 'logs' / f'{name}.log').open('x') as f:
            p = subprocess.Popen(['taskset', '-c', CORES] + a, cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT)
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
    if rec['flag'] or not rec['all_ok']:
        run_block(b, '_r1')
tel.stop.set()
(OUT / 'done.json').write_text(json.dumps(dict(finished=time.time()))); print('TIMING_DONE', flush=True)
