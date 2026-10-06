"""a100 分支：v29_timing_lib.py 的 GPU / 计时核 / 磁盘 / 背景核改为可由环境变量配置（默认值 = 3090 rev3 原值）。
A100: PANS_TIMING_GPU=6 PANS_TIMING_CORES=100-103,228-231 PANS_TIMING_DISK=sda PANS_BACKGROUND_CPUS=24,40,76,160,208,224,248
gate/telemetry 只检查计时 GPU（原来固定检查 GPU 2/3）。"""
from pathlib import Path
import subprocess
S = Path('/home/shensicheng/qyd/pans/src'); p = S / 'scripts/v29_timing_lib.py'; t = p.read_text()
def rep(a, b):
    global t
    assert t.count(a) == 1, a; t = t.replace(a, b)
rep("GPU, CORES = '2', '28-31,60-63'\nTIMING = set(range(28, 32)) | set(range(60, 64)); BACKGROUND = set(range(0, 12)) | set(range(16, 28))\nNUMA1 = set(range(16, 32)) | set(range(48, 64)); HZ = os.sysconf('SC_CLK_TCK')",
    "def _cpus(s): return {c for part in s.split(',') if part for c in (range(int(part.split('-')[0]), int(part.split('-')[-1]) + 1))}\n"
    "GPU, CORES = os.environ.get('PANS_TIMING_GPU', '2'), os.environ.get('PANS_TIMING_CORES', '28-31,60-63')\n"
    "TIMING = _cpus(CORES); BACKGROUND = _cpus(os.environ.get('PANS_BACKGROUND_CPUS', '0-11,16-27'))\n"
    "NUMA1 = _cpus(os.environ.get('PANS_NUMA_CPUS', '16-31,48-63')); HZ = os.sysconf('SC_CLK_TCK')\n"
    "DISK = os.environ.get('PANS_TIMING_DISK', 'sda'); WATCH_GPUS = tuple(os.environ.get('PANS_WATCH_GPUS', '2,3').split(','))")
rep("if l.split()[2] == 'sda'][0]", "if l.split()[2] == DISK][0]")
rep("fg = [(g, p) for g, p in gpu_procs() if g in ('2', '3') and p not in ours]", "fg = [(g, p) for g, p in gpu_procs() if g in WATCH_GPUS and p not in ours]")
rep("gp = [(g, p) for g, p in gpu_procs() if g in ('2', '3')]", "gp = [(g, p) for g, p in gpu_procs() if g in WATCH_GPUS]")
p.write_text(t)
subprocess.run(['git', '-c', 'user.email=a100@local', '-c', 'user.name=a100', 'commit', '-qam', 'A100: env-configurable timing GPU/cores/disk (defaults = rev3)'], cwd=S)
print('TIMING_LIB_PATCHED')
