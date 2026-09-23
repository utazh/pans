"""Frozen-method timing planner: disjoint query blocks, balanced order, fresh arms."""
from __future__ import annotations
from pathlib import Path
import hashlib, json, random, sys, unicodedata

def sha256(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):d.update(block)
    return d.hexdigest()

def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC',str(text)).split())

def read_bundle(path,task):
    rows=[];uids=set();texts=set();prefix=None
    for line in Path(path).read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if row.get('task')!=task or not isinstance(row.get('prefix_text'),str):
            raise ValueError('Each record needs matching task and original prefix_text')
        # Exact field used by the pans query/tokenization path.
        if not isinstance(row.get('query_text'),str) or not row['query_text']:
            raise ValueError('Missing original query_text; no fabricated replacement')
        if prefix is None:prefix=row['prefix_text']
        if row['prefix_text']!=prefix:raise ValueError('This plan is single-prefix; split workloads explicitly')
        uid=str(row['uid']);text=normalized(row['query_text'])
        if uid in uids or text in texts:raise ValueError('Duplicate UID or normalized query text')
        uids.add(uid);texts.add(text);rows.append(row)
    if not rows:raise ValueError('Empty bundle')
    return rows

def split_blocks(rows,n_blocks,size,seed):
    if n_blocks<4 or n_blocks%4 or size<1:raise ValueError('Use >=4 blocks, a multiple of four, with positive block size')
    if len(rows)<n_blocks*size:raise ValueError('Not enough UNIQUE queries; do not silently reuse them')
    ix=list(range(len(rows)));random.Random(seed).shuffle(ix)
    return [[rows[j] for j in ix[i*size:(i+1)*size]] for i in range(n_blocks)]

def balanced_orders(n_blocks,seed):
    # Four-position cyclic Latin layout, shuffled labels and block order once.
    # This balances positions, not all possible carryover effects. Fresh process per arm.
    labels=['I-original','S-original','I-reuse','S-reuse']
    rng=random.Random(seed);rng.shuffle(labels)
    base=[labels[i:]+labels[:i] for i in range(4)]
    result=[]
    for _ in range(n_blocks//4):
        this=[list(x) for x in base];rng.shuffle(this);result.extend(this)
    return result

def command(spec,folder,block,condition,arm):
    repo=Path(spec['repo']).resolve();folder=Path(folder).resolve()
    key=f'{condition}-b{block:02d}-{arm}'
    output=folder/'runs'/key
    output.relative_to(repo)  # Original runner requires an output beneath repository root.
    shared=arm.startswith('S-');admit='reuse-prefetch' if arm.endswith('reuse') else 'original'
    a=[spec.get('python',sys.executable),str(repo/'scripts/run_evidence_v26.py'),
       '--admission',admit,'--audit-jsonl',str(output)+'.v26.jsonl','--']
    for flag,k in [('--model-path','model_path'),('--store-root','store_root'),('--selector-index-dir','selector_index_dir')]:
        if spec.get(k):a.extend([flag,str(spec[k])])
    a+=['--','--pcache-root',str(spec['pcache_root'])]
    if condition!='warm':
        if condition not in ('native','preserve'):raise ValueError('Unknown cold read strategy')
        a+=['--cold','ssd-cold','--read-advice',condition,'--owned-cache-root',str(spec['pcache_root']),
            '--cold-log',str(output)+'.cold.jsonl']
    a+=['--','--mode','shared' if shared else 'identity','--backend','vllm2','--metadata-cache','--layout','hybrid',
        '--stats',str(output)+'.extra.jsonl']
    if shared:a+=['--adapters',str(spec['adapter_dir']),'--allow-candidate','--packed-ops']
    a+=['--','--task',spec['task'],'--budget',str(spec.get('budget',25)),
        '--period',str(spec.get('period',8)),'--samples',str(spec['block_size']),
        '--warmup',str(spec.get('warmup',8)),'--max-tokens','1',
        '--bundle-dir',str(folder/'bundles'/f'b{block:02d}'),'--output',str(output)]
    return a,output

def build(spec,output):
    from .guard import check,BASE
    root=Path(spec['repo']).resolve(strict=True);check(root)
    output=Path(output).resolve();output.relative_to(root)
    if output.exists():raise FileExistsError(output)
    if spec.get('method_frozen') is not True:raise ValueError('Set method_frozen=true only after freezing the method/checkpoint')
    if spec.get('admission_gpu_parity_passed') is not True:
        raise ValueError('First pass GPU parity and cache/residency audits; do not benchmark unverified admission')
    task=spec['task'];adapter=Path(spec['adapter_dir'])/(task+'.pt')
    digest=sha256(adapter)
    if digest!=spec['checkpoint_sha256']:raise ValueError('Frozen checkpoint hash mismatch')
    rows=read_bundle(spec['bundle_file'],task)
    blocks=split_blocks(rows,int(spec['blocks']),int(spec['block_size']),int(spec.get('seed',2601)))
    if not 1<=spec.get('warmup',8)<=spec['block_size']:raise ValueError('Warmup must be within block size')
    conditions=spec.get('conditions',['preserve','warm'])
    if len(conditions)!=len(set(conditions)) or any(c not in ('native','preserve','warm') for c in conditions):
        raise ValueError('Bad conditions')
    # Plan generation reads datasets for assignment only; it never uses correct/label values.
    jobs=[];output.mkdir(parents=True)
    for b,part in enumerate(blocks):
        p=output/'bundles'/f'b{b:02d}'/f'{task}.jsonl';p.parent.mkdir(parents=True)
        p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in part),encoding='utf8')
    for ci,condition in enumerate(conditions):
        orders=balanced_orders(len(blocks),spec.get('seed',2601)+ci)
        for b,order in enumerate(orders):
            for pos,arm in enumerate(order):
                argv,run=command(spec,output,b,condition,arm)
                jobs.append({'job_id':f'{condition}-b{b:02d}-{arm}', 'condition':condition,
                             'block':b,'position_in_block':pos,'arm':arm,'argv':argv,
                             'cwd':str(root),'output':str(run),
                             'bundle_sha256':sha256(output/'bundles'/f'b{b:02d}'/f'{task}.jsonl')})
    manifest={'format':'pans-frozen-evidence-study-v26','base':BASE,'spec':spec,'jobs':jobs,
              'checkpoint_sha256':digest,'source_bundle_sha256':sha256(spec['bundle_file']),
              'n_unique_queries':len(blocks)*spec['block_size'],
              'warning':'Timing queries may be historical; this is NOT a new quality holdout. '
                        'No source-proxy controls in timing. Existing cold preparation is unchanged.'}
    (output/'study.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    return manifest
