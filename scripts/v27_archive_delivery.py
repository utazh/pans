
from pathlib import Path
import hashlib,json,zipfile,time
r=Path('/home/panzihang/src/prism_layerkv')
assert (r/'audit/v27_cold_done.json').exists()
before=json.loads((r/'audit/v27_before.json').read_text())
changed=[]
for name,h in before['files'].items():
 p=r/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:changed.append(name)
orig=[]
for name,h in before['original'].items():
 p=Path(name)
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:orig.append(name)
audit={'old_files_checked':len(before['files']),'original_files_checked':len(before['original']),'changed_old':changed,'changed_original':orig,'passed':not changed and not orig}
(r/'audit/v27_final_protection.json').write_text(json.dumps(audit,indent=2));assert audit['passed'],audit
factors=[]
for p in (r/'assets').glob('v27*/*.pt'):
 hh=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):hh.update(b)
 factors.append({'path':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':hh.hexdigest(),'included_in_archive':'frozen_r0' in str(p)})
(r/'audit/v27_checkpoint_inventory.json').write_text(json.dumps(factors,indent=2))
paths=set()
for folder in ['src','scripts','tests','docs','configs','vendor']:
 for p in (r/folder).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':paths.add(p)
for folder in ['configs','audit','results','logs','figures']:
 for p in (r/folder).glob('v27*'):
  if p.is_file():paths.add(p)
  elif p.is_dir():paths.update(x for x in p.rglob('*') if x.is_file())
paths.update((r/'assets/v27_frozen_r0').glob('*.pt'))
paths.update((r/'assets/v25_g16_r4_s24').glob('*.pt'))
paths.update(p for p in (r/'upstream/pans_sharedkv_v27').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
paths.update(p for p in (r/'upstream/v27_baseline_sources').rglob('*') if p.is_file())
paths.update(p for p in (r/'data/v27').rglob('*') if p.is_file())
paths.add(r/'data/v26/p0/full_eval/trec.jsonl')
for name in ['pans_sharedkv_v27.zip','EXPERIMENT_REPORT_V27_ZH.md','README_LAYERKV_V27_SERVER.md']:paths.add(r/name)
target=r/'artifacts/pans_sharedkv_v27_server_impl.zip'
assert not target.exists()
manifest={}
with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(paths):
  rel=str(p.relative_to(r));content=p.read_bytes();manifest[rel]=hashlib.sha256(content).hexdigest();z.writestr(rel,content)
 z.writestr('MANIFEST_V27_SERVER.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(target) as z:
 assert z.testzip() is None
 for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
hh=hashlib.sha256()
with target.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):hh.update(b)
result={'path':str(target),'bytes':target.stat().st_size,'sha256':hh.hexdigest(),'files':len(manifest),'crc_and_manifest_verified':True,'excluded':'model weights, prefix stores/cache copies, full external factor/source tensors and raw logits remain on server; see checkpoint inventory'}
(r/'audit/v27_archive.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
