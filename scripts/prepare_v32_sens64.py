"""v32 sens64: 64 new train-pool queries per task for the label-free layer-sensitivity ranking (v32_layer_sens.py), disjoint from
every historical role (all data/**, frozen bundles, prism_pans data, the v29 dev set and fit512/fit64) - the same conservative
exclusion as scripts/prepare_v29_fit512.py, new seed. Prefix / labels / query layout are taken from the frozen test bundle of
the task (studies/v28_frozen/bundles/<task>_b25/eval), queries formatted with paper_tasks.format_example; --check prints one
formatted query next to a frozen test query for comparison and writes nothing. Usage: prepare_v32_sens64.py [--check]"""
import sys
sys.path.insert(0, '/home/panzihang/src/prism_layerkv/scripts')
from v26_common import *
import random
from transformers import AutoTokenizer
from contiguous_fuxian.paper_tasks import *
N, SEED, TASKS = 64, 3201, ['sst2', 'subj', 'trec', 'rte']
norm = lambda x: " ".join(x.casefold().split())
raw = Path("/home/panzihang/src/contiguous_fuxian/data/paper_tasks")
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
CHECK = '--check' in sys.argv
files = sorted({*ROOT.glob("data/**/*.jsonl"), *ROOT.glob("studies/v28_frozen/bundles/**/*.jsonl"),
                *Path("/home/panzihang/src/prism_pans/data").glob("**/*.jsonl")})
files = [f for f in files if "/data/v32/" not in str(f)]
texts = {t: [] for t in TASKS}
for f in files:
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r = json.loads(line); t = r.get("task")
        if t in texts:
            for k in ("prefix_text", "query_text"):
                if r.get(k): texts[t].append(norm(r[k]))
audit = dict(files_scanned=len(files), rule=__doc__, seed=SEED, target=N, tasks={})
if not CHECK: (ROOT / "data/v32/sens64").mkdir(parents=True, exist_ok=False)
for task in TASKS:
    blob = "\x00".join(set(texts[task]))
    template = json.loads((ROOT / f"studies/v28_frozen/bundles/{task}_b25/eval/{task}.jsonl").read_text().splitlines()[0])
    train, _ = load_task_rows(task, raw, 42)
    fields = ["premise", "hypothesis"] if task == "rte" else ["text"]
    random.Random(SEED).shuffle(train)
    if CHECK:
        print('==', task, '\nFROZEN TEST QUERY:', repr(template['query_text'][-400:]), '\nFORMATTED      :', repr((format_example(task, train[0], False) + QWEN_ASSISTANT_PREFIX)[-400:]))
        continue
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
        rows.append(dict(uid=f"{task}-v32-sens64-{len(rows)}", task=task, prefix_text=template["prefix_text"],
                         query_text=q, answer=x["label"], labels=template["labels"], label_continuation_prefix=""))
    p = ROOT / f"data/v32/sens64/{task}.jsonl"
    with p.open("x") as f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    audit["tasks"][task] = dict(stats, final=len(rows), sha256=hashlib.sha256(p.read_bytes()).hexdigest()); print(task, audit["tasks"][task], flush=True)
if not CHECK:
    with (ROOT / "audit/v32_sens64_membership.json").open("x") as f: json.dump(audit, f, indent=2)
