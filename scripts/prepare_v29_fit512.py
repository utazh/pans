"""v29 A2 fit512/fit64 (rule rev 2): 512 new train-pool queries per pilot task, disjoint from
every historical role AND the v29 dev set (same conservative exclusion as prepare_v29_dev.py).
fit64 = first 64 of fit512 in the frozen shuffled order."""
from v26_common import *
import random
from transformers import AutoTokenizer
from contiguous_fuxian.paper_tasks import *
N, SEED, TASKS = 512, 2902, ['sst2', 'rte']
norm = lambda x: " ".join(x.casefold().split())
raw = Path("/home/panzihang/src/contiguous_fuxian/data/paper_tasks")
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
files = sorted({*ROOT.glob("data/**/*.jsonl"), *ROOT.glob("studies/v28_frozen/bundles/**/*.jsonl"),
                *Path("/home/panzihang/src/prism_pans/data").glob("**/*.jsonl")})
files = [f for f in files if "/data/v29/p0/fit" not in str(f)]  # dev IS included
assert any("/data/v29/p0/dev/" in str(f) for f in files)
texts = {t: [] for t in TASKS}
for f in files:
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r = json.loads(line); t = r.get("task")
        if t in texts:
            for k in ("prefix_text", "query_text"):
                if r.get(k): texts[t].append(norm(r[k]))
audit = dict(files_scanned=len(files), rule=__doc__, seed=SEED, target=N, tasks={})
for split in ("fit512", "fit64"):
    (ROOT / f"data/v29/p0/{split}").mkdir(parents=True, exist_ok=False)
for task in TASKS:
    blob = "\x00".join(set(texts[task]))
    template = json.loads((ROOT / f"data/v26/p0/full_eval/{task}.jsonl").read_text().splitlines()[0])
    train, _ = load_task_rows(task, raw, 42)
    fields = ["premise", "hypothesis"] if task == "rte" else ["text"]
    random.Random(SEED).shuffle(train)
    stats = dict(pool=len(train), in_history=0, duplicate=0, short=0, too_long=0); seen = set(); rows = []
    for x in train:
        if len(rows) >= N: break
        vals = [norm(x[k]) for k in fields]; key = "\x01".join(vals)
        if key in seen: stats["duplicate"] += 1; continue
        seen.add(key)
        if any(v in blob for v in vals): stats["in_history"] += 1; continue
        if task == "sst2" and len(x["text"].split()) < 8: stats["short"] += 1; continue
        q = format_example(task, x, False) + QWEN_ASSISTANT_PREFIX
        if len(tok(q, add_special_tokens=False).input_ids) >= 240: stats["too_long"] += 1; continue
        rows.append(dict(uid=f"{task}-v29-fit512-{len(rows)}", task=task, prefix_text=template["prefix_text"],
                         query_text=q, answer=x["label"], labels=template["labels"], label_continuation_prefix=""))
    info = dict(stats, final=len(rows))
    for split, part in (("fit512", rows), ("fit64", rows[:64])):
        p = ROOT / f"data/v29/p0/{split}/{task}.jsonl"
        with p.open("x") as f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
        info[split + "_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    audit["tasks"][task] = info; print(task, info, flush=True)
with (ROOT / "audit/v29_fit512_membership.json").open("x") as f: json.dump(audit, f, indent=2)
