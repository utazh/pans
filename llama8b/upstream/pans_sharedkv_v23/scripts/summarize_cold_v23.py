#!/usr/bin/env python3
"""No speedup inference from one run. Lists all failures and median/min/max."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('logs',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
if Path(a.output).exists():raise FileExistsError(a.output)
out=[]
for name in a.logs:
    rows=[json.loads(s) for s in Path(name).read_text().splitlines() if s.strip()]
    valid=[r for r in rows if r['status']=='completed']
    times=[r['logits_ready_ms'] for r in valid]
    def avg(field):
        x=[r[field] for r in valid if field in r];return statistics.mean(x) if x else None
    item={'file':name,'attempts':len(rows),'completed':len(valid),
          'failures':[r for r in rows if r['status']!='completed'],
          'mean_ms':statistics.mean(times) if times else None,'median_ms':statistics.median(times) if times else None,
          'min_ms':min(times) if times else None,'max_ms':max(times) if times else None,
          'setup_mean_ms_outside_ttft':avg('setup_ms_outside_ttft'),
          'process_read_bytes_first_decoder_mean':statistics.mean([r['io_first_decoder']['read_bytes'] for r in valid]) if valid else None,
          'warning':'Process read_bytes is storage accounting; use block-device/Nsight measurements separately.'}
    out.append(item)
Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
