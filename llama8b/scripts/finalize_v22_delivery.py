import json,hashlib,zipfile,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parents[1]
for name in ('analyze_headalign_v22.py','write_layerkv_v22_report.py'):
 subprocess.run([sys.executable,'scripts/'+name],cwd=r,check=True)
u=r/'upstream/pans_sharedkv_v22';m=json.loads((u/'MANIFEST.sha256.json').read_text())
core={}
for name,h in m.items():
 if name.startswith(('src/','scripts/','tests/')):
  p=r/name
  core[name]=hashlib.sha256(p.read_bytes()).hexdigest()==h
assert all(core.values()),core
(r/'audit/v22_package_integrity.json').write_text(json.dumps({'original_package_sha256':hashlib.sha256((r/'pans_sharedkv_v22.zip').read_bytes()).hexdigest(),'installed_package_files_unchanged':core,'server_additions':'new orchestration, analysis, reports and six Qwen geometry CUDA tests only'},indent=2))
readme="""# v2.2 server experiment delivery

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
"""
(r/'README_LAYERKV_V22_SERVER.md').write_text(readme)
paths=set()
for pattern in ('src/contiguous_fuxian/sharedkv_v22/*.py','scripts/*v22*.py','scripts/v22*.json','tests/test_headalign_v22*.py','assets/v22_*/*','results/v22*.json','results/v22*.csv','results/v22*.jsonl','results/v22*/summary.json','results/v22*/scored_records.jsonl','results/v22*/command.json','logs/v22*.log','logs/v22*.jsonl','audit/v22*.json','README*V22*.md','EXPERIMENT_REPORT_V22_ZH.md','upstream/pans_sharedkv_v22/docs/*.md','upstream/pans_sharedkv_v22/NOTICE.md'):
 paths.update(p for p in r.glob(pattern) if p.is_file())
paths.discard(r/'audit/v22_delivery_manifest.json')
paths.discard(r/'logs/v22_delivery.log')  # This log remains live until this process exits.
manifest={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
mp=r/'audit/v22_delivery_manifest.json';mp.write_text(json.dumps(manifest,indent=2));paths.add(mp)
out=r/'artifacts/pans_sharedkv_v22_server_impl.zip'
temporary=out.with_suffix('.zip.tmp')
assert not temporary.exists(),temporary
with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(paths):z.write(p,str(p.relative_to(r)))
with zipfile.ZipFile(temporary) as z:assert z.testzip() is None
temporary.replace(out)
assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
print('DELIVERY',out,'bytes',out.stat().st_size,'files',len(paths))
