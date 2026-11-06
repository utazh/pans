
from pathlib import Path
import json,re,random,hashlib,shutil,sys
from transformers import AutoTokenizer
r=Path('/home/panzihang/src/prism_layerkv');out=r/'data/v24_splits'
assert not out.exists();out.mkdir()
old=json.loads((r/'data/v23_splits/fit/trec.jsonl').read_text().splitlines()[0])
norm=lambda s:' '.join(s.casefold().split())
question=lambda s:re.search(r'Question: (.*?)\nType:',s,re.S).group(1)
excluded=set();prior_queries=set();prefixes=set();past={}
for p in (r/'data').rglob('*.jsonl'):
 if p.is_relative_to(out):continue
 b=p.read_bytes();past[str(p.relative_to(r))]=hashlib.sha256(b).hexdigest()
 for line in b.decode('utf8').splitlines():
  row=json.loads(line)
  if row.get('task')!='trec':continue
  prior_queries.add(row['query_text']);excluded.add(norm(question(row['query_text'])))
  prefixes.add(row['prefix_text'])
for prefix in prefixes:
 for q in re.findall(r'Question: (.*?)\nType:',prefix,re.S):excluded.add(norm(q))
rawpath=Path('/home/panzihang/src/contiguous_fuxian/data/paper_tasks/raw/train_5500.label')
raw=rawpath.read_bytes()
try:text=raw.decode('utf8');encoding='utf8'
except UnicodeDecodeError:text=raw.decode('latin1');encoding='latin1'
pool=[];seen=set()
for index,line in enumerate(text.splitlines()):
 label,q=line.split(' ',1);key=norm(q)
 if key in excluded or key in seen:continue
 seen.add(key);pool.append((index,q,label.split(':')[0]))
random.Random(20260924).shuffle(pool)
assert len(pool)>=1536
tokenizer=AutoTokenizer.from_pretrained('/data1/llm/Qwen/Qwen2.5-7B-Instruct',local_files_only=True)
def digest(s):
 ids=tokenizer(s,add_special_tokens=False).input_ids
 return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
oldhash={digest(s) for s in prior_queries};hashes={};counts={}
for split,items in [('dev768',pool[:768]),('final768',pool[768:1536])]:
 directory=out/split;directory.mkdir()
 rows=[]
 for raw_index,q,label in items:
  row=dict(old,uid=f'trec-v24-{split}-{raw_index}',answer=label)
  row['query_text']=re.sub(r'Question: .*?\nType:',lambda _:f'Question: {q}\nType:',old['query_text'],count=1,flags=re.S)
  rows.append(row)
 hashes[split]={x['uid']:digest(x['query_text']) for x in rows}
 assert len(set(hashes[split].values()))==768 and not set(hashes[split].values())&oldhash
 (directory/'trec.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
 counts[split]=len(rows)
for split in ['fit','val','test500','dev']:
 (out/split).mkdir();shutil.copy2(r/f'data/v23_splits/{split}/trec.jsonl',out/split/'trec.jsonl')
 rows=[json.loads(x) for x in (out/split/'trec.jsonl').read_text().splitlines()]
 hashes[split]={x['uid']:digest(x['query_text']) for x in rows};counts[split]=len(rows)
keys=list(hashes)
for i,a in enumerate(keys):
 for b in keys[i+1:]:assert not set(hashes[a].values())&set(hashes[b].values()),(a,b)
meta={'seed':20260924,'counts':counts,'query_hashes':hashes,'new_dev_final_disjoint_from_all_prior_bundle_queries':True,'prior_bundle_files':past,'excluded_normalized_questions':len(excluded),'remaining_eligible_train_pool':len(pool),'raw_sha256':hashlib.sha256(raw).hexdigest(),'raw_encoding':encoding,'final768_status':'untouched_until_all_topology_and_checkpoint_choices_frozen','fit_val':'Exact v23 fit64/val64 reused for matched objective/seed ablations','dev128':'Historical descriptive regression only; not stage admission','test500':'Historically observed official test regression; not model selection'}
(r/'audit/v24_data_isolation.json').write_text(json.dumps(meta,indent=2))
print(json.dumps({k:v for k,v in meta.items() if k not in ['query_hashes','prior_bundle_files']},indent=2))
