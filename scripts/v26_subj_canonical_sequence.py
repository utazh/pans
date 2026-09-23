from v26_common import *
import signal
qpid=10329
done=ROOT/'audit/v26_p1_rte_b25_quality_S_process.json'
while not done.exists():
 if 'Traceback' in (ROOT/'logs/v26_quality_gpu3.log').read_text():raise RuntimeError('GPU3 quality failed')
 time.sleep(5)
assert json.loads(done.read_text())['returncode']==0
assert not (ROOT/'audit/v26_p0_trec_b5_frozen.json').exists(),'Quality already advanced; do not overlap'
a=Path(f'/proc/{qpid}/cmdline').read_bytes().split(b'\0')
assert b'scripts/v26_quality_worker.py' in a and a[a.index(b'--gpu')+1]==b'3'
os.kill(qpid,signal.SIGSTOP)
try:
 assert not (ROOT/'audit/v26_p0_trec_b5_frozen.json').exists(),'Quality advanced during pause'
 uuid=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert uuid not in apps,apps
 print('Reserved idle GPU3 for corrected SUBJ; quality worker paused before next model launch',flush=True)
 for pref,budgets in [('p0','5,10,25,50'),('p1','25')]:
  run('v26_subj_canonical_'+pref+'_training_worker',[PY,'-u','scripts/train_subj_canonical_v26.py','--gpu','3','--prefix',pref,'--budgets',budgets],3)
 with (ROOT/'audit/v26_subj_canonical_training_done.json').open('x') as f:json.dump(dict(status='completed',gpu=3,readers=5,quality_worker_resumed=qpid),f)
finally:
 os.kill(qpid,signal.SIGCONT)
 print('Resumed GPU3 quality worker',flush=True)
