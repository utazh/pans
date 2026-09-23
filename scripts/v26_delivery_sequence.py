
from v26_common import *
while not (ROOT/'audit/v26_diagnostics_done.json').exists():
 if 'Traceback' in (ROOT/'logs/v26_diagnostics_sequence2.log').read_text():raise RuntimeError('Diagnostics failed; no incomplete delivery')
 time.sleep(10)
for name,script in [('v26_primary_analysis','analyze_primary_v26.py'),('v26_matrix_analysis','analyze_v26_results.py'),('v26_nsight_analysis','analyze_v26_nsight.py'),('v26_final_invariants','check_v26_invariants.py'),('v26_plot','plot_v26_results.py'),('v26_report','write_v26_report.py')]:
 run(name,[PY,'-u','scripts/'+script],3)
# Packaging is a separate final action after report/figure review.
with (ROOT/'audit/v26_report_ready.json').open('x') as f:json.dump(dict(status='ready_for_agent_review'),f)
