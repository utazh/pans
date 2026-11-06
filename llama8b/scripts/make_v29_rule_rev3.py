"""Rule rev 3: timing telemetry/cores under the known persistent background load (user decision
2026-09-29: proceed without cleanup). Only timing_telemetry / timing_design.cores change."""
import hashlib, json, shutil, time
from pathlib import Path
R = Path('/home/panzihang/src/prism_layerkv')
cur = R / 'configs/v29_selection_rule.json'
rule = json.loads(cur.read_text()); assert rule['revision'] == 2
old = json.loads(json.dumps(rule))
rule['revision'] = 3
rule['timing_environment'] = dict(
    background_load=('Unauthorized persistent process (/tmp/mysqld, 24 threads each pinned to one CPU: 0-11 and 16-27) '
                     'present on the host, respawned via a PAM hook on every login/sudo/cron session. The user decided on '
                     '2026-09-29 to measure without cleanup. Historical timings (prism_pans rounds, v26-v28) were very likely '
                     'taken under the same load. Absolute TTFT is therefore host-condition specific; comparisons are paired '
                     'within blocks under identical conditions.'),
    cores='28-31,60-63: the 4 NUMA1 physical cores (plus HT siblings) not used by the background threads; all arms identical')
rule['timing_telemetry'] = dict(
    exclusive='GPU2, cores 28-31,60-63; no other job of this user anywhere on the host',
    pre_start_gate=['nvidia-smi: no compute process on GPU2 or GPU3',
                    'timing cores 28-31,60-63 TOTAL busy < 1 core averaged over 30 s (/proc/stat)',
                    'sda TOTAL read+write < 5 MB/s over 30 s (/proc/diskstats)'],
    recorded_not_gated='total busy on background CPUs 0-11,16-27 and the rest of NUMA1 (to document the load level)',
    in_run_sampling=rule['timing_telemetry']['in_run_sampling'],
    block_flag=('attributable evidence only: a foreign compute process on GPU2/GPU3, or processes outside the timing '
                'process tree last run on the timing cores using >= 1 core over any 30 s window'),
    flagged_block=rule['timing_telemetry']['flagged_block'], never=rule['timing_telemetry']['never'])
changed = sorted(k for k in set(rule) | set(old) if rule.get(k) != old.get(k))
assert set(changed) <= {'revision', 'timing_environment', 'timing_telemetry'}, changed
shutil.copy(cur, R / 'configs/v29_selection_rule.rev2.json')
with (R / 'configs/v29_selection_rule.rev3.tmp').open('x') as f: json.dump(rule, f, indent=1)
shutil.move(R / 'configs/v29_selection_rule.rev3.tmp', cur)
fz = json.loads((R / 'audit/v29_rule_freeze.json').read_text())
h = hashlib.sha256(cur.read_bytes()).hexdigest()
fz['previous'].append(dict(revision=2, sha256=fz['sha256'], file='configs/v29_selection_rule.rev2.json'))
fz.update(revision=3, sha256=h, rev3_unix=time.time(), rev3_changed=changed,
          rev3_reason='Background load makes the rev2 NUMA1 gate unsatisfiable; user chose to measure without cleanup. '
                      'Timing cores moved to the 4 NUMA1 cores free of background threads; gate/flag scoped to those cores. '
                      'Selection (A1/A2) content unchanged; no timing block had started under rev2.')
(R / 'audit/v29_rule_freeze.json').write_text(json.dumps(fz, indent=1))
print('rev3', h, changed)
