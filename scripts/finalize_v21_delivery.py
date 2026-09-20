#!/usr/bin/env python3
from pathlib import Path
import json,shutil,hashlib,difflib,os,zipfile
ROOT=Path(__file__).resolve().parents[1];orig=ROOT.with_name('prism_pans')
for rel in ('runtime_tools/nsight-2025.5.1','runtime_deps'):
 dst=ROOT/rel
 if not dst.exists():
  dst.parent.mkdir(parents=True,exist_ok=True)
  shutil.copytree(orig/rel,dst,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
p=ROOT/'scripts/profile_layerkv_v21.py';s=p.read_text().replace("sys.path.insert(0,'/home/panzihang/src/prism_pans/runtime_deps')","sys.path.insert(0,str(ROOT/'runtime_deps'))");p.write_text(s)
p=ROOT/'scripts/v21_gpu2_next.sh';s=p.read_text().replace('NSYS=/home/panzihang/src/prism_pans/runtime_tools','NSYS=/home/panzihang/src/prism_layerkv/runtime_tools')
s=s.replace('while kill -0 26987 2>/dev/null; do sleep 5; done',"""while pgrep -u "$USER" -f '^[^ ]*/python( -u)? scripts/run_layerkv_v21_suite.py scripts/v21_quality_jobs.json$' >/dev/null; do sleep 5; done""");p.write_text(s)
for before_name,base,out in [('source_before.json',orig,'v21_original_integrity.json'),('v21_before.json',ROOT,'v21_prior_experiment_integrity.json')]:
 before=json.loads((ROOT/'audit'/before_name).read_text());changed=[str(p) for p,h in before.items() if not (base/p).exists() or hashlib.sha256((base/p).read_bytes()).hexdigest()!=h]
 report={'checked_files':len(before),'changed_files':changed,'unchanged':not changed}
 (ROOT/'audit'/out).write_text(json.dumps(report,indent=2));assert not changed,report
cut=(ROOT/'pans_sharedkv_v21.zip').stat().st_mtime;changed=[]
for path,dirs,files in os.walk(orig):
 for name in files:
  p=Path(path)/name
  if p.stat().st_mtime>cut:changed.append(str(p.relative_to(orig)))
(ROOT/'audit/v21_original_mtime.json').write_text(json.dumps({'cutoff_unix':cut,'files_modified_after_upload':changed,'none_modified':not changed,'scope':'Files, not directory access/mtime metadata. Two new Nsight bytecode caches were removed; see cleanup audit.'},indent=2))
assert not changed,changed
up=ROOT/'upstream/pans_sharedkv_v21';patch=[]
for p in sorted(up.rglob('*.py')):
 dst=ROOT/p.relative_to(up)
 if dst.exists() and p.read_text()!=dst.read_text():patch.extend(difflib.unified_diff(p.read_text().splitlines(True),dst.read_text().splitlines(True),fromfile='upstream/'+str(p.relative_to(up)),tofile=str(p.relative_to(up))))
(ROOT/'audit/v21_package_fixes.patch').write_text(''.join(patch))
from datetime import datetime,timezone
manifest={'completed_utc':datetime.now(timezone.utc).isoformat(),'original_files_unchanged':True,'prior_v1_code_unchanged':True,'profiler_tools_now_isolated':True,'code':{}}
paths=set((ROOT/'src/contiguous_fuxian/sharedkv_v21').glob('*.py'))|set((ROOT/'scripts').glob('*v21*'))|set((ROOT/'tests').glob('*v21*.py'))
for p in sorted(paths):
 if p.is_file():manifest['code'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'audit/v21_delivery_manifest.json').write_text(json.dumps(manifest,indent=2))
p=ROOT/'EXPERIMENT_REPORT_V21_ZH.md';s=p.read_text().replace('原 prism_pans 和上一版实验代码保持不变。','原 prism_pans 的源码和原有文件内容未修改；Nsight 帮助命令自动生成的两个字节码缓存已清理。上一版实验代码未变。')
s+='\nProfiler 工具已独立复制到 runtime_tools/nsight-2025.5.1 和 runtime_deps；后续 v21 wrapper 只从新目录调用。清理记录见 audit/v21_nsight_side_effect_cleanup.json。\n'
p.write_text(s)
p=ROOT/'README_LAYERKV_V21.md';s=p.read_text().replace('## Profiling','## Profiling\n\nProfiler 工具的独立副本在本工程 runtime_tools/nsight-2025.5.1、runtime_deps；v21 wrapper 后续不再从原工程导入工具。');p.write_text(s)
artifact=ROOT/'artifacts/pans_sharedkv_v21_server_impl.zip';artifact.parent.mkdir(exist_ok=True)
assert not artifact.exists()
with zipfile.ZipFile(artifact,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(paths):
  if p.is_file():z.write(p,str(p.relative_to(ROOT)))
 for p in [ROOT/'README_LAYERKV_V21.md',ROOT/'EXPERIMENT_REPORT_V21_ZH.md',*sorted((ROOT/'audit').glob('v21*.json')),*sorted((ROOT/'audit').glob('v21*.patch'))]:
  z.write(p,str(p.relative_to(ROOT)))
 for name in ('v21_hold_compare.json','v21_hold_paired_controls.json','v21_reg_compare.json','v21_timing_summary.json','v21_nsight_summary.json','v21_viz_summary.json','v21_invariants.json'):
  p=ROOT/'results'/name;z.write(p,str(p.relative_to(ROOT)))
print('DELIVERY_READY',artifact,'bytes',artifact.stat().st_size)
