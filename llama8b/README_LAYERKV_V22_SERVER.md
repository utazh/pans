# v2.2 server experiment delivery

Full results and decisions: EXPERIMENT_REPORT_V22_ZH.md.
All experiment paths are under /home/panzihang/src/prism_layerkv on [server address omitted].
Original prism_pans and prior code remain untouched.

This is an additive overlay for the audited v2 interfaces. See README_HEADALIGN_V22.md for package use. The server directory has no Git metadata; all eight expected interface blob hashes matched before installation (audit/v22_before.json). Runtime guards remain enabled.

Python: /home/panzihang/venvs/vllm-stable/bin/python.
Use PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=src, OMP_NUM_THREADS=4 and a free CUDA_VISIBLE_DEVICES.
Set TMPDIR=/home/panzihang/src/prism_layerkv/runtime_data/tmp_v22.
Do not rerun existing output joblists blindly: the entrypoints reject overwrites.
Use new output paths for any rerun.

- Compile lists: scripts/screen_headalign_v22.py and train_headalign_v22.py
- Exact evaluation/control/timing arguments: scripts/v22_*_jobs.json
- Suite runner: scripts/run_layerkv_v22_suite.py
- Profilers: scripts/profile_suite_v22.py (isolated Nsight and VizTracer under this directory)
- Final reports: results/v22_*summary.json, v22_analysis.json, v22_quality_compare.json, v22_fixed_compare.json, v22_invariants.json
- Tests: logs/v22_all_tests.log
- Raw profile files: profiles/v22/ (kept on server, excluded from compact ZIP)
- Raw per-query logits: results/v22_*_logits/ (kept on server, excluded from compact ZIP)

All readers are diagnostic candidates. No local gate was relaxed. The 175 TREC requests are previously observed regression data, not an unseen final test.
