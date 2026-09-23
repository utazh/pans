"""Small, explicit contracts; no guessed paths or overwrite."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
BASE='11fde71c36ceecded60d560620900dec5c3a5d76'
V25_FORMAT='pans-shared-prefix-native-suffix-v25'
FIELDS=('task','token_hash','prefix_tokens','geometry','period','keep_ratio','model_identity')
def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def json_new(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def torch_new(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:torch.save(value,f)
def load_tensor(path):
 return torch.load(path,map_location='cpu',weights_only=True)
def validate_sources(sources,period=None):
 n=len(sources)
 if not n:raise ValueError('Empty source map')
 for lid,s in enumerate(sources):
  if type(s) is not int or not 0<=s<n or sources[s]!=s:raise ValueError('Only one-hop owned source maps')
  if period is not None:
   if period<1 or lid//period!=s//period or (lid%period==0 and lid!=s):
    raise ValueError('Same selector window; native leaders required')
