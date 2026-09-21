from pathlib import Path
import json,hashlib,zipfile,time
r=Path(__file__).resolve().parents[1]
assert (r/'EXPERIMENT_REPORT_V23_ZH.md').exists()
assert json.loads((r/'audit/v23_final_invariants.json').read_text())['all_passed']
paths=set()
for directory,pattern in [('scripts','*v23*.py'),('tests','test_sharedkv_v23*.py'),('configs','*v23*.json'),('audit','v23*'),('logs','v23*'),('results','v23*'),('assets','v23*'),('figures','v23*')]:
 for p in (r/directory).glob(pattern):
  candidates=p.rglob('*') if p.is_dir() else [p]
  for q in candidates:
   if q.is_file() and '__pycache__' not in q.parts and 'delivery' not in q.name:paths.add(q)
for directory in ['src/contiguous_fuxian/sharedkv_v23','data/v23_splits','upstream/pans_sharedkv_v23']:
 paths.update(p for p in (r/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
paths.update([r/'EXPERIMENT_REPORT_V23_ZH.md',r/'README_LAYERKV_V23_SERVER.md'])
profile=[]
for p in (r/'profiles/v23').rglob('*'):
 if p.is_file():
  profile.append({'path':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  if p.suffix=='.json' and p.name!='viztracer.json':paths.add(p)
pp=r/'audit/v23_profile_inventory.json';pp.write_text(json.dumps(profile,indent=2));paths.add(pp)
paths={p for p in paths if p.name not in ['v23_delivery_manifest.json','v23_delivery_result.json']}
files={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
manifest={'created_unix':time.time(),'server_root':str(r),'files':files,'scope':'Server overlay, fixed reader candidates, raw request data, results, logs, audit, plot. Large raw profiler files stay on server; inventory includes hashes. Existing v2.2 guarded base is required.'}
mp=r/'audit/v23_delivery_manifest.json';mp.write_text(json.dumps(manifest,indent=2));paths.add(mp)
out=r/'artifacts/pans_sharedkv_v23_server_impl.zip';tmp=out.with_suffix('.zip.tmp')
assert not out.exists() and not tmp.exists()
with zipfile.ZipFile(tmp,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(paths):z.write(p,str(p.relative_to(r)))
with zipfile.ZipFile(tmp) as z:
 assert z.testzip() is None
 for name,sha in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha,name
tmp.replace(out)
for name,sha in files.items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==sha,name
result={'archive':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'archived_files':len(paths),'manifest_files_verified':len(files)}
(r/'audit/v23_delivery_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
