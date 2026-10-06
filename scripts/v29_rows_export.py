"""Unified row-engine data, step 1: per-layer contiguous FP16 K/V files assembled from the ACTUAL pcache payload the
pcache path restores (runtime_data/v29_pcache_gpu2_ssd), not re-converted from the BF16 store.
pcache naming (HyperInfer KV_Division): key chunk c -> <pid>_<layer>_<c>_k.npy ; value chunk c -> <pid>_<layer>_<c+cc>_v.npy,
cc = ceil(N/16); each file is a raw [<=16, 4, 128] FP16 buffer. Output: <out>/<task>/L<layer>_{k,v}.f16 raw [N,4,128] FP16.
Checks: every assembled chunk is byte-identical to its pcache file; additionally reports the max |diff| against the BF16
prefix store cast to FP16 (informational). Usage: v29_rows_export.py <out_dir>"""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np, torch
R = Path('/home/shensicheng/qyd/pans/src'); PC = R / 'runtime_data/v29_pcache_gpu2_ssd'
OUT = Path(sys.argv[1]); TASKS = ['sst2', 'subj', 'trec', 'rte']  # --store-tasks order == pid
OUT.mkdir(parents=True, exist_ok=False); report = {}
for pid, task in enumerate(TASKS):
    meta = json.loads((R / f'runtime_data/prefix_store/{task}/metadata.json').read_text())
    n, h, d = meta['prefix_tokens'], meta['kv_heads'], meta['head_dim']; cc = math.ceil(n / 16); row = h * d * 2
    (OUT / task).mkdir(); rep = dict(prefix_tokens=n, chunks=cc, max_abs_vs_store_fp16=0.0, files={})
    for layer in range(meta['layers']):
        for kind, off in (('k', 0), ('v', cc)):
            buf = bytearray()
            for c in range(cc):
                p = PC / f'{pid}_{layer}_{c + off}_{kind}.npy'; b = p.read_bytes()
                rows = min(16, n - 16 * c)
                if len(b) != rows * row: raise SystemExit(f'size mismatch {p}: {len(b)} != {rows * row}')
                buf += b
            dst = OUT / task / f'L{layer:02d}_{kind}.f16'
            with dst.open('xb') as f: f.write(buf)
            back = dst.read_bytes()
            for c in range(cc):  # byte-identity with every pcache chunk
                a, b2 = back[c * 16 * row:(c * 16 + min(16, n - 16 * c)) * row], (PC / f'{pid}_{layer}_{c + off}_{kind}.npy').read_bytes()
                if a != b2: raise SystemExit(f'byte mismatch {task} L{layer} {kind} chunk {c}')
            st = np.fromfile(R / f'runtime_data/prefix_store/{task}/layer_{layer:02d}_{"key" if kind == "k" else "value"}.bf16', dtype=np.uint16)
            ref = torch.from_numpy(st.astype(np.int16)).view(torch.bfloat16).reshape(n, h, d).to(torch.float16).float()
            got = torch.from_numpy(np.frombuffer(back, dtype=np.float16).reshape(n, h, d).copy()).float()
            rep['max_abs_vs_store_fp16'] = max(rep['max_abs_vs_store_fp16'], float((ref - got).abs().max()))
            rep['files'][dst.name] = hashlib.sha256(back).hexdigest()
    report[task] = rep
    print(task, n, 'chunks', cc, 'max|pcache-store_fp16|', rep['max_abs_vs_store_fp16'], flush=True)
(OUT / 'manifest.json').write_text(json.dumps(dict(source_pcache=str(PC), layout='raw [N,4,128] float16, logical token order',
                                                   chunk_byte_identity='verified for every chunk', tasks=report), indent=1))
