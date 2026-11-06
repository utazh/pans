
from v26_common import *
import zipfile
assert (ROOT/'EXPERIMENT_REPORT_V26_ZH.md').is_file()
assert (ROOT/'audit/v26_diagnostics_done.json').is_file()
target=ROOT/'artifacts/pans_sharedkv_v26_server_impl.zip';assert not target.exists()
files=set()
for top in ['src','vendor','tests','scripts','configs','docs']:
 for p in (ROOT/top).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.so']:files.add(p)
for top in ['audit','logs','results','assets']:
 for p in (ROOT/top).rglob('*'):
  if p.is_file() and any('v26' in part.lower() for part in p.relative_to(ROOT).parts) and p.suffix not in ['.pt','.sqlite','.nsys-rep']:
   files.add(p)
for p in (ROOT/'assets').glob('v26*/*.pt'):files.add(p)
files.add(ROOT/'assets/v25_g16_r4_s24/trec.pt')
for p in (ROOT/'data/v26').rglob('*'):
 if p.is_file():files.add(p)
for p in (ROOT/'profiles/v26').rglob('*'):
 if p.is_file() and p.name in ['validated.json','command.json','profile_requests.json']:files.add(p)
files.update([ROOT/'EXPERIMENT_REPORT_V26_ZH.md',ROOT/'README_LAYERKV_V26_SERVER.md',ROOT/'pans_sharedkv_v26.zip'])
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
 z.writestr('V26_ARCHIVE_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(target) as z:
 assert z.testzip() is None
 for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
result=dict(path=str(target),bytes=target.stat().st_size,files=len(manifest)+1,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),verified=True,excluded='Model weights, physical prefix/cache stores, teacher tensors and raw profiler traces remain on server; archive includes derived evidence, scripts, evaluation/calibration bundles and exported readers.')
with (ROOT/'audit/v26_archive.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
