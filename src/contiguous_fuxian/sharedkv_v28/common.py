from __future__ import annotations
import hashlib,json,os
from pathlib import Path
BASE='681fe495028598e9d51f2f57220ecb96ed0e5f35'
FIELDS=('task','token_hash','prefix_tokens','geometry','period','keep_ratio','model_identity')
TASKS={'sst2':872,'subj':1003,'trec':500,'rte':277}
BUDGETS=(5,10,25,50)
CHECKS={
 'src/contiguous_fuxian/sharedkv_v27/frozen_control.py':'b4b645bdcbeafd73ee2ba95c7139510a822f59d3',
 'src/contiguous_fuxian/sharedkv_v27/baseline_math.py':'128eac940ddbb4bf31b5bd5fb0945133d6a23f4b',
 'src/contiguous_fuxian/sharedkv_v27/baseline_quality.py':'060cda19cd6017465d5f47422920c901509973b0',
 'scripts/run_baseline_quality_v27.py':'8eb22a474be2e1199f4722148e7f7baa1437e427',
 'scripts/run_evidence_v26.py':'64076b8085e21ef0f8b84134b7a4b812fedfcad4',
 'scripts/v27_search_kvsharer.py':'07c5b40943cf2a1fdd69ad4c75e3a22b1abf8376',
 'scripts/v27_bf16_sequence.py':'56f76ca263f9988dda2547ef827a1bbb05fd68ec',
}
def check(repo):
 for name,expected in CHECKS.items():
  data=(Path(repo)/name).read_bytes()
  actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  if actual!=expected: raise ValueError('Audited interface changed: '+name)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write_json(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def save_tensor(path,value):
 import torch
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:torch.save(value,f)
 # Server-only lossless artifact encoding for resident quality references.
 # No change to tensors or theoretical/deployment byte accounting.
 if isinstance(value,dict) and value.get('method')=='direct-map-prefix-quality':
  import zipfile,shutil
  temporary=p.with_suffix(p.suffix+'.deflate')
  with zipfile.ZipFile(p,'r') as source,zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as target:
   for item in source.infolist():
    with source.open(item) as a,target.open(item.filename,'w',force_zip64=True) as b:
     shutil.copyfileobj(a,b,1<<20)
  check_value=torch.load(temporary,map_location='cpu',weights_only=True)
  for lid,kv in value['native'].items():
   for a,b in zip(kv,check_value['native'][lid]):
    if a.dtype!=b.dtype or not torch.equal(a,b):raise ValueError('Lossless artifact verification failed')
  temporary.replace(p)
def load_tensor(path):
 import torch
 return torch.load(path,map_location='cpu',weights_only=True)
def read_jsonl(path):
 with Path(path).open(encoding='utf8') as f:return [json.loads(s) for s in f if s.strip()]
def write_jsonl(path,rows):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def tree_files(root):return {str(p.resolve()):sha(p) for p in sorted(Path(root).rglob('*.pt'))}
def assert_hashes(mapping):
 for p,h in mapping.items():
  if sha(p)!=h:raise ValueError('Frozen input changed: '+p)
def compare_identity(a,b):
 for key in FIELDS:
  if a[key]!=b[key]:raise ValueError('Different identity: '+key)
def require_cell(c):
 if c['task'] not in TASKS or c['budget'] not in BUDGETS:raise ValueError('Unknown task/budget')
 if c['expected_n']!=TASKS[c['task']]:raise ValueError('Keep the frozen full evaluation split size')
 for name in ('r4','teacher_fit','teacher_val','eval_file','validation_file'):
  if not Path(c[name]).exists():raise FileNotFoundError(f"{c['id']} missing explicit {name}: {c[name]}")
 if c['id']!=f"{c['task']}_b{c['budget']}":raise ValueError('Cell id differs from task/budget')
