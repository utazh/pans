from pathlib import Path
import hashlib,json
r=Path('/home/panzihang/src/prism_layerkv')
p=r/'scripts/run_round4.py'
s=p.read_text()
s=s.replace("p.add_argument('--task',", "p.add_argument('--bundle-dir',type=Path,default=ROOT/'data/paper_task_bundles_full_eval_strict')\np.add_argument('--max-tokens',type=int,default=1)\np.add_argument('--task',",1)
s=s.replace("str(ROOT/'data/paper_task_bundles_full_eval_strict')","str(a.bundle_dir)")
s=s.replace("'--max-tokens','1'","'--max-tokens',str(a.max_tokens)")
p.write_text(s)
p=r/'src/contiguous_fuxian/sharedkv/core.py';s=p.read_text().replace('Literal["reference", "flash2"]','Literal["reference", "flash2", "vllm2"]')
needle='    if backend == "flash2":'
block='''    if backend == "vllm2":
        if q_positions is not None or q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("vllm2 requires CUDA fp16/bf16 with contiguous query positions")
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        b, tq, hq, d = q.shape
        nk, hk = k.shape[1:3]
        # No repeated GQA KV heads or reconstructed consumer prefix.
        cuq = torch.arange(0, (b+1)*tq, tq, dtype=torch.int32, device=q.device)
        cuk = torch.arange(0, (b+1)*nk, nk, dtype=torch.int32, device=q.device)
        out, lse = flash_attn_varlen_func(
            q.reshape(b*tq,hq,d), k.reshape(b*nk,hk,d), v.reshape(b*nk,hk,d),
            cuq, cuk, tq, nk, causal=causal, softmax_scale=scale,
            return_softmax_lse=True, fa_version=2)
        return out.reshape_as(q), lse.reshape(hq,b,tq).permute(1,0,2)
'''
s=s.replace(needle,block+needle);p.write_text(s)
p=r/'src/contiguous_fuxian/sharedkv/runtime.py';s=p.read_text().replace('("reference","flash2")','("reference","flash2","vllm2")').replace('if self.backend=="flash2":','if self.backend in ("flash2","vllm2"):').replace('backend="flash2",causal=True','backend=self.backend,causal=True');p.write_text(s)
p=r/'scripts/run_sharedkv.py';s=p.read_text().replace('["reference","flash2"]','["reference","flash2","vllm2"]')
start=s.index('    checks={'); end=s.index('    for name,expected',start)
checks={}
for name in ['scripts/run_round4.py','src/contiguous_fuxian/flexgen_qwen_reprefill.py','src/contiguous_fuxian/flexgen_pcache.py']:
    b=(r/name).read_bytes();checks[name]=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
s=s[:start]+'    checks='+repr(checks)+'\n'+s[end:];p.write_text(s)
(r/'audit/integration_hashes.json').write_text(json.dumps(checks,indent=2))
# Reproducible request-level split, with the same immutable prefix.
import random
split_uids={}
for task in ['trec','sst2','subj','rte']:
    rows=[json.loads(x) for x in (r/'data/paper_task_bundles_full_eval_strict'/f'{task}.jsonl').read_text().splitlines()]
    random.Random(20260918).shuffle(rows)
    splits={'fit':rows[:32], 'val':rows[32:64], 'test':rows[64:192]}
    split_uids[task]={k:[x['uid'] for x in v] for k,v in splits.items()}
    for split,rs in splits.items():
        d=r/'data/sharedkv_splits'/split;d.mkdir(parents=True,exist_ok=True)
        (d/f'{task}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rs))
(r/'data/sharedkv_splits/manifest.json').write_text(json.dumps(split_uids,indent=2))
print('Adapted backend and launcher; splits 32/32/128')
