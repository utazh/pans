from pathlib import Path
import hashlib,json
r=Path('/home/panzihang/src/prism_layerkv')
p=r/'scripts/run_round4.py';s=p.read_text().replace("str(a.warmup),","str(max(1,a.warmup)),");p.write_text(s)
p=r/'scripts/run_sharedkv.py';s=p.read_text().replace('p.add_argument("--stats",type=Path)','p.add_argument("--stats",type=Path)\n    p.add_argument("--logits",type=Path)')
needle='    sys.argv=[str(ROOT/"scripts/run_round4.py"),*argv]'
block='''    if a.logits:
        import torch
        from contiguous_fuxian.sharedkv.io import query_hash
        a.logits.mkdir(parents=True,exist_ok=True)
        original_completion = runner.greedy_flexgen_completion
        def capture(*args, **kw):
            output = kw.get("first_token_logits_out")
            if output is None:
                output = []
                kw["first_token_logits_out"] = output
            result = original_completion(*args, **kw)
            torch.save({"logits":output[-1], "query_hash":query_hash(kw["query_token_ids"]),
                        "selected_by_layer":kw["loader"]._selected_by_layer},
                       a.logits / (query_hash(kw["query_token_ids"])+".pt"))
            return result
        runner.greedy_flexgen_completion = capture
'''
s=s.replace(needle,block+needle)
start=s.index('    checks=');end=s.index('    for name,expected',start)
checks={}
for name in ['scripts/run_round4.py','src/contiguous_fuxian/flexgen_qwen_reprefill.py','src/contiguous_fuxian/flexgen_pcache.py']:
 b=(r/name).read_bytes();checks[name]=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
s=s[:start]+'    checks='+repr(checks)+'\n'+s[end:];p.write_text(s)
(r/'audit/integration_hashes.json').write_text(json.dumps(checks,indent=2))
