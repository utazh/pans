import json, glob, collections
B = '/home/shensicheng/qyd/pans/'
for f in ['src/data/v26/p0/fit/trec.jsonl', 'src/data/v29/p0/dev/trec.jsonl', 'src/studies/v28_frozen/bundles/trec_b25/eval/trec.jsonl',
          'src/studies/v28_frozen/bundles/trec_b25/b00/trec.jsonl', 'work/q7/bundles_strict/trec.jsonl']:
    try:
        r = json.loads(open(B + f).readline())
    except Exception as e:
        print(f, e); continue
    print(f, {k: (v[:70] + '...' if isinstance(v, str) and len(v) > 70 else v) for k, v in r.items() if k != 'labels'})
c = collections.Counter(); im = collections.Counter()
for f in glob.glob(B + 'src/data/**/*', recursive=True) + glob.glob(B + 'src/studies/v28_frozen/bundles/**/*', recursive=True) \
        + glob.glob(B + 'work/q7/bundles_*/**/*', recursive=True):
    name = f.rsplit('/', 1)[-1]
    if '.' in name:
        ext = name.rsplit('.', 1)[-1]; c[ext] += 1
        if ext in ('json', 'jsonl', 'txt'):
            t = open(f, encoding='utf-8', errors='replace').read()
            if '<|im_' in t: im[ext] += 1
print('ext', dict(c)); print('files containing <|im_', dict(im))
