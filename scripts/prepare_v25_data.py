
from pathlib import Path
import json,re,random,hashlib,shutil
from transformers import AutoTokenizer
r=Path(__file__).resolve().parents[1];out=r/'data/v25_splits';out.mkdir(exist_ok=False)
template=json.loads((r/'data/v24_splits/fit/trec.jsonl').read_text().splitlines()[0])
norm=lambda s:' '.join(s.casefold().split())
excluded=set();prior_queries=set();prefixes=set();past={}
for p in (r/'data').rglob('*.jsonl'):
 if p.is_relative_to(out):continue
 b=p.read_bytes();past[str(p.relative_to(r))]=hashlib.sha256(b).hexdigest()
 for line in b.decode('utf8').splitlines():
  row=json.loads(line)
  if row.get('task')!='trec':continue
  prior_queries.add(row['query_text'])
  excluded.add(norm(re.search(r'Question: (.*?)\nType:',row['query_text'],re.S).group(1)))
  prefixes.add(row['prefix_text'])
for prefix in prefixes:
 for q in re.findall(r'Question: (.*?)\nType:',prefix,re.S):excluded.add(norm(q))
raw=Path('/home/panzihang/src/contiguous_fuxian/data/paper_tasks/raw/train_5500.label').read_bytes()
try:text=raw.decode('utf8');encoding='utf8'
except UnicodeDecodeError:text=raw.decode('latin1');encoding='latin1'
pool=[];seen=set()
for idx,line in enumerate(text.splitlines()):
 label,q=line.split(' ',1);key=norm(q)
 if key in excluded or key in seen:continue
 seen.add(key);pool.append((idx,q,label.split(':')[0]))
random.Random(20260925).shuffle(pool);assert len(pool)>=1536,len(pool)
tok=AutoTokenizer.from_pretrained('/data1/llm/Qwen/Qwen2.5-7B-Instruct',local_files_only=True)
def digest(text):return hashlib.sha256(json.dumps(tok(text,add_special_tokens=False).input_ids,separators=(',',':')).encode()).hexdigest()
oldhash={digest(q) for q in prior_queries};hashes={};rows=[]
for idx,q,label in pool[:1536]:
 row=dict(template,uid=f'trec-v25-final1536-{idx}',answer=label)
 row['query_text']=re.sub(r'Question: .*?\nType:',lambda _:f'Question: {q}\nType:',template['query_text'],count=1,flags=re.S)
 rows.append(row)
(out/'final1536').mkdir();(out/'final1536/trec.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
hashes['final1536']={x['uid']:digest(x['query_text']) for x in rows}
assert len(set(hashes['final1536'].values()))==1536 and not set(hashes['final1536'].values())&oldhash
for dest,src in [('explore768','dev768'),('test500','test500'),('fit','fit'),('val','val')]:
 (out/dest).mkdir();shutil.copy2(r/f'data/v24_splits/{src}/trec.jsonl',out/dest/'trec.jsonl')
 rows=[json.loads(x) for x in (out/dest/'trec.jsonl').read_text().splitlines()]
 hashes[dest]={x['uid']:digest(x['query_text']) for x in rows}
keys=list(hashes)
for i,a in enumerate(keys):
 for b in keys[i+1:]:assert not set(hashes[a].values())&set(hashes[b].values()),(a,b)
meta={'seed':20260925,'query_hashes':hashes,'counts':{k:len(v) for k,v in hashes.items()},'prior_bundle_files':past,'raw_sha256':hashlib.sha256(raw).hexdigest(),'raw_encoding':encoding,'remaining_eligible_train_pool':len(pool),'excluded_normalized_questions':len(excluded),'final1536_status':'new_train_pool_holdout_unseen_until_structure_and_seeds_frozen','explore768_status':'observed_v24_dev_reused_only_for_exploratory_selection','test500_status':'historical_regression_only','no_overlap_final_with_all_prior_bundle_queries':True}
(r/'audit/v25_data_isolation.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k not in ['query_hashes','prior_bundle_files']},indent=2))
