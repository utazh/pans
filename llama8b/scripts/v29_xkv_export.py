"""Export xKV-CLF/P8 BF16 factors (v28 representation, one pre-set rank (256,384)) for SSD-native restoration.
Per cell: <out>/<cell>/g<group>_<kind>.u16  raw row-major [N, rank] BF16 bits (the token-indexed shared basis U,
                                            stored on SSD and read row-wise at request time)
          <out>/<cell>/resident.pt           per-layer right factors (rank x H*D, BF16), lookup, rope_theta, geometry
                                            (resident on GPU, accounted like our readers)
          <out>/<cell>/manifest.json         shapes, bytes, SHA-256, source representation SHA.
Usage: v29_xkv_export.py <out_dir> <cell> [<cell> ...]"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch
R = Path('/home/shensicheng/qyd/pans/src')
REP = 'clf_256_384_bf16'
if '--rep' in sys.argv:
    i = sys.argv.index('--rep'); REP = sys.argv[i + 1]; del sys.argv[i:i + 2]
out = Path(sys.argv[1]); cells = sys.argv[2:]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for cell in cells:
    src = R / f'studies/v28_frozen/representations/{cell}/{REP}.pt'
    b = torch.load(src, map_location='cpu', weights_only=False)
    assert b['method'] == 'xKV-CLF-prefix-quality' and b['factor_precision'] == 'torch.bfloat16'
    d = out / cell; d.mkdir(parents=True, exist_ok=False)
    files, resident = {}, dict(task=b['task'], token_hash=b['token_hash'], prefix_tokens=b['prefix_tokens'], geometry=b['geometry'],
                               keep_ratio=b['keep_ratio'], rope_theta=b['rope_theta'], position_offset=b['position_offset'],
                               lookup=b['lookup'], groups=[])
    for gi, g in enumerate(b['groups']):
        item = dict(layers=g['layers'])
        for kind in ('key', 'value'):
            left = g[kind]['left'].contiguous(); n, r = left.shape
            p = d / f'g{gi}_{kind}.u16'
            with p.open('xb') as f: f.write(left.view(torch.int16).numpy().tobytes())
            files[p.name] = dict(shape=[n, r], bytes=p.stat().st_size, sha256=sha(p))
            item[kind] = dict(file=p.name, rows=n, rank=r, rights=[x.contiguous() for x in g[kind]['rights']])
        resident['groups'].append(item)
    torch.save(resident, d / 'resident.pt')
    rb = sum(x.numel() * x.element_size() for g in resident['groups'] for k in ('key', 'value') for x in g[k]['rights'])
    (d / 'manifest.json').write_text(json.dumps(dict(source=str(src), source_sha256=sha(src), files=files,
        ssd_bytes=sum(f['bytes'] for f in files.values()), resident_right_factor_bytes=rb,
        per_token_ssd_bytes=sum(f['shape'][1] * 2 for f in files.values()),
        variant=f'xKV-CLF/P8-SSD: v28 {REP}, shared U on SSD, per-layer rights resident'), indent=1))
    print(cell, 'ssd MB %.1f resident MB %.1f' % (sum(f['bytes'] for f in files.values()) / 1e6, rb / 1e6), flush=True)
