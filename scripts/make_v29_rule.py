"""Write configs/v29_selection_rule.json (DRAFT until status == 'frozen')."""
import json, hashlib
from pathlib import Path

ROOT = Path('/home/panzihang/src/prism_layerkv')
orders = json.loads(Path('/tmp/v29_revert_orders.json').read_text())
dev_audit = json.loads((ROOT / 'audit/v29_dev_membership.json').read_text())

cells = {}
for cell, o in orders.items():
    consumers = sorted(l for l, s in enumerate(o['sources']) if s != l)
    order = o['revert_order']
    cells[cell] = dict(
        r0_checkpoint=f'studies/v28_frozen/readers/{cell}/{cell.split("_")[0]}.pt',
        r0_sha256=o['r0_sha256'],
        r0_native_blocks=o['blocks'],
        parent_sources=o['sources'],
        revert_order=order,
        candidates={
            'I': dict(kept_consumers=[], physical_sources=28),
            'G24': dict(kept_consumers=sorted(set(consumers) - set(order[:8])), physical_sources=24),
            'G20': dict(kept_consumers=sorted(set(consumers) - set(order[:4])), physical_sources=20),
            'G16': dict(kept_consumers=consumers, physical_sources=16),
        })

import sys
STATUS = sys.argv[1] if len(sys.argv) > 1 else 'DRAFT'
BYTES_DEFINITION = Path('/tmp/v29_bytes_definition.txt').read_text().strip()
INIT_DEFINITION = Path('/tmp/v29_init_definition.txt').read_text().strip()

rule = {
    'format': 'pans-v29-selection-rule',
    'revision': int(sys.argv[2]) if len(sys.argv) > 2 else 1,
    'status': STATUS,
    'frozen_before': 'Any A1 shared-candidate evaluation on the dev set (identity dev runs are allowed before freezing: they are candidate-independent).',
    'dev_set': dict(path='data/v29/p0/dev', per_task=500,
                    file_sha256={t: v['file_sha256'] for t, v in dev_audit['tasks'].items()},
                    membership_audit='audit/v29_dev_membership.json',
                    role='configuration screening only; not a quality certification set'),
    'candidates': dict(
        set=['I', 'G24', 'G20', 'G16'],
        construction=('Frozen v28 R0 checkpoint (R4-trained weights, native_policy.blocks=0). '
                      'G20/G24 keep a subset of consumers via sharedkv_v21.bundles.isolate(bundle, kept_consumers); '
                      'reverted consumers read their own prefix KV. No reader value changes, no retraining, '
                      'no new sources. validate_sources (one-hop, same P8 window, leaders native) must pass.'),
        revert_order_rule=('Label-free, fixed before evaluation: sort consumers by descending |l - s(l)|, '
                           'ties by ascending layer index. G20 reverts the first 4, G24 the first 8.'),
        not_monotonic_warning='Reverting consumers changes downstream hidden states and online P8 selections; every candidate is evaluated end to end.'),
    'eligibility': dict(
        metric='paired net correct-count difference on the same dev queries: correct(candidate) - correct(I)',
        threshold=('correct(candidate) - correct(I) >= -ceil(0.01 * n); for n = 500 at most 5 net fewer '
                   'correct answers. Integer counts only; no rounded percentages.'),
        reported_as='percentage points (1 pp = 0.01 absolute accuracy)',
        identity_always_eligible=True),
    'selection': dict(
        objective='minimize bytes among eligible candidates',
        bytes_field='summary.json mean_physical_prefetch_kv_bytes',
        bytes_definition=BYTES_DEFINITION,
        missing_bytes='a candidate whose bytes field is missing or non-finite is invalid for selection; never treated as 0; pause and revise',
        tie_rule=('Let B_min be the smallest bytes among eligible candidates. Among eligible candidates with '
                  '(B - B_min) / B_min < 0.005 choose, in order: larger net correct-count difference; fewer '
                  'shared consumers; fixed name order I, G24, G20, G16, then any A2 candidate by name.'),
        no_eligible_shared='Select I and report the cell as having no qualified compression point.',
        recorded_not_used=['prediction disagreement rate vs I', 'harmful/beneficial flips vs I',
                           'mean KL between candidate-label score distributions (softmax over mean label '
                           'log-prob, T=1) vs I', 'I accuracy vs gold']),
    'a2': dict(
        data_size_pilot=dict(
            cells=['sst2_b5', 'rte_b5'],
            arms=['fit64', 'fit512'],
            data=('fit512: 512 new queries per task, disjoint from dev and all historical roles, built with the '
                  'dev-set exclusion rule (seed 2902). fit64: the first 64 of fit512 in the frozen shuffled order, '
                  'so fit64 is a subset of fit512 and both arms have identical historical exposure.'),
            initialization=INIT_DEFINITION,
            fixed=('same source map (G16), seed, validation set (existing v26 val traces), optimizer, loss, '
                   'checkpoint-selection rule; both arms perform exactly 512 query updates (fit64: 8 passes, '
                   'fit512: 1 pass); train with R4, infer with frozen R0'),
            sampling=('rev 2: epoch-shuffled sampling WITHOUT replacement for both arms (the legacy trainer samples '
                      'with replacement via rng.randrange, which would cover only ~63% of fit512 in 512 steps). '
                      'Implemented by scripts/train_v29_epoch.py, which replaces random.Random in '
                      'sharedkv_v24.train with a seeded epoch-permutation generator; legacy code unchanged. '
                      'Each pass is a fresh seeded permutation; fit64 = exactly 8 passes, fit512 = exactly 1 pass.'),
            teacher=('teacher records captured with the same identity capture command as v26 '
                     '(collect_teacher_v26.py, budget of the cell, P8), separately for fit64 and fit512; '
                     'plan = v26 cell plan sources with design_query_hashes = the arm\'s own fit hashes'),
            switch=('Engineering switch, not a significance claim: enable recalibration only if '
                    'correct(fit512) - correct(fit64) >= ceil(0.01 * n) (>= 5 of 500) on the dev set, R0 inference, '
                    'in BOTH pilot cells. Compared arm vs arm under identical training conditions; not vs the old '
                    '200-step checkpoint and not only vs I.'),
            pilot_results_are_candidates=('The pilot fit512 checkpoints (and fit64) are A2 candidates for their '
                                          'cells and go through the same eligibility/selection rule; they are not '
                                          'retrained if the switch fires.'),
            record=['processed tokens', 'train wall time', 'dev net correct-count difference', 'bytes']),
        recalibration=dict(
            max_extra_candidates_per_cell=1,
            max_calibration_runs_per_candidate=1,
            trigger='cell (other than the two pilots) whose selected configuration after A1 is I or G24',
            recipe='G16 structure retrained with the fit512 recipe, only if the pilot switch fired; otherwise no recalibration',
            same_rule='each A2 candidate is judged once by this eligibility/selection rule; no further tuning on dev')),
    'freeze_after_a2': ('Record final configuration and checkpoint SHA-256 per cell, then rerun full 16-cell test-set '
                        'quality for the selected configurations (plus I) before any timing. Old G16 accuracy is never '
                        'paired with a different configuration\'s TTFT.'),
    'timing_design': dict(
        arms=('I, G16 (frozen v28 R0), and the selected configuration if its checkpoint or execution config '
              'differs from both; a retrained G16 is a different arm from the frozen G16'),
        ssd_cold_preserve=dict(cells='all 16', blocks=6, queries_per_block=16),
        ordinary_cache=dict(cells='4 tasks x {25%, 50%} = 8 cells, fixed in advance by budget, not by results',
                            blocks=6, queries_per_block=16,
                            claim_limit='not a full ordinary-cache matrix'),
        order='each block runs all arms; 6 blocks cover all 6 orders of 3 arms (Latin/complete balance); 2 arms alternate',
        max_runs=432,
        analysis='paired by block; report per-block direction; no significance guarantee from 6 blocks'),
    'timing_telemetry': dict(
        exclusive='GPU2, cores 16-31,48-63; no other job of this user anywhere on the host',
        pre_start_gate=[
            'nvidia-smi: no compute process on GPU2 or GPU3',
            'NUMA1 (cores 16-31,48-63) TOTAL busy < 2 cores averaged over a 30 s window (/proc/stat; total, not attributed)',
            'sda TOTAL read+write < 5 MB/s over a 30 s window (/proc/diskstats; total, not attributed)'],
        in_run_sampling=('every 5 s: nvidia-smi compute processes per GPU; per-process CPU time deltas from '
                         '/proc/<pid>/stat for every process (attributable: owner, pid, cpus_allowed); '
                         '/proc/<pid>/io for processes of this user; totals from /proc/stat and /proc/diskstats '
                         'recorded but not used for attribution'),
        block_flag=('flag only on attributable evidence: a foreign compute process on GPU2/GPU3, or processes outside '
                    'the timing process tree using >= 2 cores of NUMA1 CPU time over any 30 s window. I/O of other '
                    'users cannot be attributed without root: recorded as unknown, never used to drop or flag blocks '
                    'by subtraction of counters'),
        flagged_block='keep and mark; rerun the whole paired block (all arms, same queries, same order) once; '
                      'if the rerun is also flagged, report both and mark the cell interfered',
        never='drop or rerun blocks based on observed TTFT'),
    'change_control': dict(
        semantic=('changes to candidates, thresholds, byte definition, missing-field handling, data membership or '
                  'validity: pause affected parts, keep the old rule, publish a new revision number and SHA-256, '
                  'rerun every affected comparison under one rule (never only the unfavorable arm)'),
        non_semantic=('path fixes, extra logging, equivalent parsing fixes: record code diff and passing tests, '
                      'continue; unaffected results are not rerun')),
    'reporting': dict(
        main_table='accuracy and change vs I per cell; Full-KV per task listed as a reference (never called an upper bound)',
        companion_table='disagreement rate and label-score KL vs I',
        wording=('Task accuracy measures answer quality; disagreement and KL measure fidelity to the no-sharing sparse '
                 'system; the two can diverge (applies to all tasks). SUBJ/25% is highly sensitive to the selection '
                 'reuse period: on the dev set, P4 recovers no-sharing accuracy (P8 57.4, P4 85.0, P1 86.8); '
                 'only 25% was tested.')),
    'cells': cells,
}
out = ROOT / (sys.argv[3] if len(sys.argv) > 3 else 'configs/v29_selection_rule.json')
with out.open('x') as f:
    json.dump(rule, f, indent=1)
print(out, hashlib.sha256(out.read_bytes()).hexdigest())
