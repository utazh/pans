"""Create and verify a server-only reproducibility overlay."""
from pathlib import Path
import hashlib,json,zipfile
r=Path(__file__).resolve().parents[1];files=set()
for directory in ['src','vendor','tests','scripts']:
 files.update(p for p in (r/directory).rglob('*.py') if '__pycache__' not in p.parts)
for pattern in ['configs/v25*.json','configs/v25_explore/*.json','docs/sharedkv_v25_package/*','audit/v25*','results/v25*','assets/v25*','logs/v25*.log','logs/v25*.jsonl','figures/v25*','data/v25_splits','runtime_data/v24_teacher_fit','runtime_data/v24_teacher_val','assets/v24_score_g26_s24','configs/v24_ladder/*.json' ]:
 for p in r.glob(pattern):
  if p.is_dir():files.update(x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts)
  elif p.is_file():files.add(p)
files.update(r/n for n in ['EXPERIMENT_REPORT_V25_ZH.md','README_LAYERKV_V25_SERVER.md'])
# Do not archive a growing delivery log or a self-referential archive checksum.
files={p for p in files if p.name not in ['v25_delivery.log','v25_archive_validation.json','v25_delivery_manifest.json'] and p.suffix not in ['.pyc']}
profiles=[{'path':str(p.relative_to(r)),'bytes':p.stat().st_size} for p in sorted((r/'profiles/v25').rglob('*')) if p.is_file()]
(r/'audit/v25_raw_profile_inventory.json').write_text(json.dumps(profiles,indent=2));files.add(r/'audit/v25_raw_profile_inventory.json')
manifest={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)}
mp=r/'audit/v25_delivery_manifest.json';mp.write_text(json.dumps(manifest,indent=2));files.add(mp)
out=r/'artifacts/pans_sharedkv_v25_server_impl.zip';assert not out.exists()
with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(files):z.write(p,str(p.relative_to(r)))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for name,meta in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==meta['sha256'],name
report={'path':str(out),'files':len(files),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'all_manifest_members_verified':True,'raw_profiles_included':False,'local_experiment_files_written':False}
(r/'audit/v25_archive_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
