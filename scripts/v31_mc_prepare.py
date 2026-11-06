"""v31 many-class many-shot ICL data (Banking77, CLINC150, TREC-fine) in the v26 bundle format, one shared prefix per dataset.
Raw files: data/v31_mc/raw (sha256 recorded). Storage slots (names only, content-agnostic engine, as for QuALITY):
  banking77 -> sst2, clinc150 -> subj, trecfine -> trec.
Prompt (Qwen chat, same layout as the v26 TREC prompt): one user turn = instruction + full label list + k shuffled
demonstrations per class; the query ends with '<|im_end|>\\n<|im_start|>assistant\\n'; label scoring = candidate label
continuation log-likelihood over all class names.
Splits (seed 42, disjoint, all from the official train split): demos k/class, fit 64, val 64, dev 500; test = official
test split (CLINC: in-scope only; Banking77 / CLINC: fixed stratified subset of 1000, reported as a subset; TREC: all 500).
Pre-declared shots: Banking77 k=4, CLINC150 k=3, TREC-fine k=6 (capped by class availability).
Usage: v31_mc_prepare.py   (system python3 for parquet; tokenizer check via the vLLM venv afterwards)"""
import hashlib, json, random, re
from collections import defaultdict
from pathlib import Path
R = Path('/home/panzihang/src/prism_layerkv'); RAW = R / 'data/v31_mc/raw'; OUT = R / 'data/v31_mc'
SLOT = {'banking77': 'sst2', 'clinc150': 'subj', 'trecfine': 'trec'}
SHOTS = {'banking77': 4, 'clinc150': 3, 'trecfine': 6}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
nice = lambda s: s.replace('_', ' ')

def load():
    d = {}
    tr = [json.loads(l) for l in (RAW / 'banking77_train.jsonl').read_text().splitlines()]
    te = [json.loads(l) for l in (RAW / 'banking77_test.jsonl').read_text().splitlines()]
    d['banking77'] = ([(r['text'], nice(r['label_text'])) for r in tr], [(r['text'], nice(r['label_text'])) for r in te])
    lines = (RAW / 'clinc_README.md').read_text().splitlines(); names = {}
    i = next(k for k, l in enumerate(lines) if 'class_label' in l)
    for l in lines[i + 2:]:
        m = re.match(r"\s*'(\d+)': (.+)$", l)
        if not m: break
        names[int(m.group(1))] = m.group(2).strip()
    assert len(names) == 151 and 'oos' in names.values()
    def cl(split):
        # clinc_plus_<split>.jsonl = row-for-row JSONL export of the official parquet (pandas/pyarrow, off-server; parquet sha in audit)
        rs = [json.loads(l) for l in (RAW / f'clinc_plus_{split}.jsonl').read_text().splitlines()]
        return [(r['text'], nice(names[int(r['intent'])])) for r in rs if names[int(r['intent'])] != 'oos']
    d['clinc150'] = (cl('train') + cl('validation'), cl('test'))
    def tr_(p):
        out = []
        for l in (RAW / p).read_text(encoding='latin-1').splitlines():
            lab, q = l.split(' ', 1); out.append((q.strip(), lab))
        return out
    d['trecfine'] = (tr_('train_5500.label'), tr_('TREC_10.label'))
    return d

HEAD = {'banking77': ('Classify each banking customer query into one of the following intents.', 'Query', 'Intent'),
        'clinc150': ('Classify each user request into one of the following intents.', 'Request', 'Intent'),
        'trecfine': ('Classify each question into one of the following fine-grained question types (COARSE:fine).', 'Question', 'Type')}

audit = dict(raw={p.name: sha(p) for p in sorted(RAW.iterdir()) if p.is_file()}, datasets={})
for name, (train, test) in load().items():
    rng = random.Random(42); labels = sorted({y for _, y in train} | {y for _, y in test})
    by = defaultdict(list)
    for i, (x, y) in enumerate(train): by[y].append(i)
    demos = []
    for y in labels:
        idx = by[y][:]; rng.shuffle(idx); demos += idx[:min(SHOTS[name], len(idx))]
    rng.shuffle(demos); used = set(demos)
    rest = [i for i in range(len(train)) if i not in used]; rng.shuffle(rest)
    fit, val, dev = rest[:64], rest[64:128], rest[128:628]
    if name == 'trecfine': tst = list(range(len(test)))
    else:   # fixed stratified subset of 1000: round-robin over classes in a seeded order
        byt = defaultdict(list)
        for i, (_, y) in enumerate(test): byt[y].append(i)
        for y in byt: rng.shuffle(byt[y])
        tst, k = [], 0
        while len(tst) < 1000:
            for y in labels:
                if k < len(byt[y]) and len(tst) < 1000: tst.append(byt[y][k])
            k += 1
        tst.sort()
    instr, xk, yk = HEAD[name]
    prefix = '<|im_start|>user\n' + instr + '\n\n' + 'Labels: ' + ', '.join(labels) + '.\n\n' + \
             ''.join(f'{xk}: {train[i][0]}\n{yk}: {train[i][1]}\n\n' for i in demos)
    slot = SLOT[name]
    def rows(split, src, ids):
        return [dict(uid=f'{name}-v31-{split}-{j}', task=slot, prefix_text=prefix,
                     query_text=f'{xk}: {src[i][0]}\n{yk}:\n\n<|im_end|>\n<|im_start|>assistant\n',
                     answer=src[i][1], labels=labels, label_continuation_prefix='') for j, i in enumerate(ids)]
    for split, src, ids in (('fit', train, fit), ('val', train, val), ('dev', train, dev), ('test', test, tst)):
        d = OUT / split; d.mkdir(parents=True, exist_ok=True)
        p = d / f'{slot}.jsonl'
        if p.exists(): raise SystemExit(f'exists: {p}')
        p.write_text(''.join(json.dumps(r) + '\n' for r in rows(split, src, ids)))
    (OUT / 'eval').mkdir(exist_ok=True)
    (OUT / 'eval' / f'{slot}.jsonl').write_text((OUT / 'dev' / f'{slot}.jsonl').read_text())   # prefix-store build input
    audit['datasets'][name] = dict(slot=slot, classes=len(labels), shots=SHOTS[name], demos=len(demos), prefix_chars=len(prefix),
                                   train_pool=len(train), test_pool=len(test), fit=len(fit), val=len(val), dev=len(dev), test=len(tst),
                                   test_is_subset=name != 'trecfine', demo_ids_sha=hashlib.sha256(json.dumps(sorted(demos)).encode()).hexdigest(),
                                   disjoint=len(set(demos) | set(fit) | set(val) | set(dev)) == len(demos) + 64 + 64 + 500)
(OUT / 'prepare_audit.json').write_text(json.dumps(audit, indent=1)); print(json.dumps(audit['datasets'], indent=1))
