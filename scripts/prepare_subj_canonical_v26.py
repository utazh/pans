from v26_common import *
import re
from transformers import AutoTokenizer
raw=Path('/home/panzihang/src/contiguous_fuxian/data/paper_tasks/subj');canonical={}
for name,label in [('plot.tok.gt9.5000','objective'),('quote.tok.gt9.5000','subjective')]:
 lines=(raw/name).read_bytes().decode('latin-1').split('\n');assert len([x for x in lines if x])==5000
 for line in lines:
  if line:canonical.setdefault(' '.join(line.split()),set()).add(label)
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);audit={}
for pref in ['p0','p1']:
 sets={};fixes={}
 for split in ['fit','val','eval','full_eval']:
  original=ROOT/f'data/v26/{pref}/{split}/subj.jsonl';rows=[json.loads(x) for x in original.read_text().splitlines()];changes=[]
  for row in rows:
   sentence=row['query_text'].split('Sentence: ',1)[1].split('\nSubjectivity:')[0];key=' '.join(sentence.split())
   if row['answer'] not in canonical.get(key,set()):
    candidates=[text for text,labs in canonical.items() if key in text and row['answer'] in labs]
    assert len(candidates)==1,(row['uid'],len(candidates));whole=candidates[0]
    changes.append(dict(uid=row['uid'],fragment=sentence,canonical=whole))
    row['query_text']=row['query_text'].replace('Sentence: '+sentence+'\nSubjectivity:','Sentence: '+whole+'\nSubjectivity:',1)
   if split in ['fit','val']:assert len(tok(row['query_text'],add_special_tokens=False).input_ids)<240
  keys={' '.join(x['query_text'].split('Sentence: ',1)[1].split('\nSubjectivity:')[0].split()) for x in rows}
  assert len(keys)==len(rows);sets[split]=keys;fixes[split]=changes
  target=ROOT/f'data/v26/{pref}/canonical_{split}';target.mkdir(exist_ok=False)
  with (target/'subj.jsonl').open('x') as f:f.write(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
 prefix=rows[0]['prefix_text'];demos={' '.join(t.split()) for t,l in re.findall(r'Sentence: (.*?)\nSubjectivity: (subjective|objective)\n',prefix,re.S)}
 assert not(sets['fit']&sets['val'] or sets['fit']&sets['full_eval'] or sets['val']&sets['full_eval'])
 assert all(not value&demos for value in sets.values())
 audit[pref]=dict(repairs=fixes,counts={k:len(v) for k,v in sets.items()},canonical_query_disjoint=True,prefix_unchanged=True)
with (ROOT/'audit/v26_subj_canonical_correction.json').open('x') as f:json.dump(dict(reason='Legacy latin1 str.splitlines treats 21 CP1252 ellipsis bytes 0x85 as boundaries. Reconnect fragments to unique physical-LF source records, preserve frozen membership/UIDs and prefix. Custom evaluation remains 1003; not a standard 1000 split.',raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.glob('*.5000')},prefixes=audit),f,indent=2)
print('SUBJ corrected: 2 fit, 1 validation, 2 full-eval fragments per prefix; data roles remain disjoint.')
