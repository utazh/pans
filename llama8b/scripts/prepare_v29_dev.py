"""v29 development set: 500 queries/task from the train pools, disjoint from every
historical data role (all data/*.jsonl incl. eval/timing/fit/val splits, prism_pans
data, v28 bundles) and from few-shot demos in every prefix seen in those files.
Exclusion is conservative: any candidate text field appearing anywhere in the
concatenated historical texts is rejected. Not a certification set."""
from v26_common import *
import random
from transformers import AutoTokenizer
from contiguous_fuxian.paper_tasks import *
N,SEED=500,2901
norm=lambda x:" ".join(x.casefold().split())
raw=Path("/home/shensicheng/qyd/pans/work/data_raw")
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
files=sorted({*ROOT.glob("data/**/*.jsonl"),*ROOT.glob("studies/v28_frozen/bundles/**/*.jsonl"),*Path("/home/panzihang/src/prism_pans/data").glob("**/*.jsonl")})
files=[f for f in files if "/data/v29/" not in str(f)]
texts={t:[] for t in ["sst2","subj","trec","rte"]};subj_used=set()
for f in files:
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():continue
        r=json.loads(line);t=r.get("task")
        if t not in texts:continue
        for k in ("prefix_text","query_text"):
            if r.get(k):texts[t].append(norm(r[k]))
        if t=="subj" and "Sentence: " in r.get("query_text",""):
            subj_used.add(norm(r["query_text"].split("Sentence: ",1)[1].split("\nSubjectivity:")[0]))
out=ROOT/"data/v29/p0/dev";out.mkdir(parents=True,exist_ok=False);audit=dict(files_scanned=len(files),rule=__doc__,seed=SEED,target=N,tasks={})
for task in texts:
    blob="\x00".join(set(texts[task]))
    tfile=ROOT/(f"data/v26/p0/canonical_full_eval/subj.jsonl" if task=="subj" else f"data/v26/p0/full_eval/{task}.jsonl")
    template=json.loads(tfile.read_text().splitlines()[0])
    train,_=load_task_rows(task,raw,42)
    if task=="subj":
        canon={}
        for name,label in [("plot.tok.gt9.5000","objective"),("quote.tok.gt9.5000","subjective")]:
            for line in (raw/"subj"/name).read_bytes().decode("latin-1").split("\n"):
                if line:canon.setdefault(" ".join(line.split()),set()).add(label)
        train=[dict(text=s,label=next(iter(l))) for s,l in canon.items() if len(l)==1]
    fields=["premise","hypothesis"] if task=="rte" else ["text"]
    random.Random(SEED).shuffle(train)
    stats=dict(pool=len(train),in_history=0,duplicate=0,short=0,too_long=0,subj_contains_used=0);seen=set();rows=[]
    for x in train:
        if len(rows)>=N:break
        vals=[norm(x[k]) for k in fields];key="\x01".join(vals)
        if key in seen:stats["duplicate"]+=1;continue
        seen.add(key)
        if any(v in blob for v in vals):stats["in_history"]+=1;continue
        if task=="subj" and any(u in vals[0] for u in subj_used if len(u)>=15):stats["subj_contains_used"]+=1;continue
        if task=="sst2" and len(x["text"].split())<8:stats["short"]+=1;continue
        q=format_example(task,x,False)+QWEN_ASSISTANT_PREFIX
        if len(tok(q,add_special_tokens=False).input_ids)>=240:stats["too_long"]+=1;continue
        rows.append(dict(uid=f"{task}-v29-dev-{len(rows)}",task=task,prefix_text=template["prefix_text"],query_text=q,answer=x["label"],labels=template["labels"],label_continuation_prefix=""))
    p=out/f"{task}.jsonl"
    with p.open("x") as f:f.write("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    lab={};[lab.__setitem__(r["answer"],lab.get(r["answer"],0)+1) for r in rows]
    audit["tasks"][task]=dict(stats,final=len(rows),labels=lab,prefix_sha256=hashlib.sha256(template["prefix_text"].encode()).hexdigest(),file_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),historical_texts=len(texts[task]))
    print(task,audit["tasks"][task],flush=True)
with (ROOT/"audit/v29_dev_membership.json").open("x") as f:json.dump(audit,f,indent=2)
