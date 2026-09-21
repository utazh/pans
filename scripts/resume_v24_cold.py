from pathlib import Path
import json,time,subprocess,os
r=Path(__file__).resolve().parents[1];py='/home/panzihang/venvs/vllm-stable/bin/python'
while not (r/'results/v24_cold_fresh_probe/summary.json').exists():
 if 'Traceback (most recent call last)' in (r/'logs/v24_cold_fresh_probe.log').read_text():raise RuntimeError('Fresh-process cold probe failed')
 time.sleep(5)
z=[json.loads(x) for x in (r/'results/v24_cold_fresh_probe.cold.jsonl').read_text().splitlines()]
assert len(z)==2 and all(x['status']=='completed' and x['page_after']['resident']==0 and x['application_payload_empty'] for x in z)
d=json.loads((r/'audit/v24_cold_restart.json').read_text());d['fresh_process_probe_passed']=True;d['probe_requests']=2
(r/'audit/v24_cold_restart.json').write_text(json.dumps(d,indent=2))
a=json.loads((r/'configs/v24_formal_timing_jobs.json').read_text())[8:]
with (r/'configs/v24_formal_cold_jobs.json').open('x') as f:json.dump(a,f,indent=2)
print('FRESH PROCESS PROBE PASSED; RESUME ALL 16 COLD ARMS',flush=True)
subprocess.run([py,'-u','scripts/run_v24_cold_jobs.py',str(r/'configs/v24_formal_cold_jobs.json')],cwd=r,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True)
print('COLD FORMAL COMPLETE',flush=True)
