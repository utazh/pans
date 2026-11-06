"""v29 A1 candidates: G20/G24 derived from frozen v28 R0 via sharedkv_v21.bundles.isolate.
Reader values are untouched; only reverted consumers become native. Writes one adapter
dir per candidate (runtime loads <dir>/<task>.pt)."""
from pathlib import Path
import hashlib, json, sys
import torch
ROOT = Path('/home/shensicheng/qyd/pans/src')
sys.path.insert(0, str(ROOT / 'src'))
from contiguous_fuxian.sharedkv_v21.bundles import isolate

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rule_path = ROOT / 'configs/v29_selection_rule.json'
rule = json.loads(rule_path.read_text())
assert rule['status'] == 'frozen' and rule['revision'] == 1
rule_sha = sha(rule_path)
assert rule_sha == json.loads((ROOT / 'audit/v29_rule_freeze.json').read_text())['sha256']

out_root = ROOT / 'studies/v29/readers'
out_root.mkdir(parents=True, exist_ok=False)
manifest = {}
for cell, c in rule['cells'].items():
    task = cell.split('_')[0]
    src = ROOT / c['r0_checkpoint']
    assert sha(src) == c['r0_sha256'], cell
    bundle = torch.load(src, map_location='cpu', weights_only=False)
    assert bundle['native_policy']['blocks'] == 0 and bundle['sources'] == c['parent_sources']
    manifest[cell] = {'G16': dict(adapter_dir=str(src.parent), sha256=c['r0_sha256'], derived=False)}
    for name in ('G20', 'G24'):
        kept = c['candidates'][name]['kept_consumers']
        child = isolate(bundle, kept)
        n_src = len(set(child['sources']))
        assert n_src == c['candidates'][name]['physical_sources'], (cell, name, n_src)
        for lid, s in enumerate(child['sources']):
            if s != lid:
                assert torch.equal(child['readers'][str(lid)]['A'], bundle['readers'][str(lid)]['A'])
        child['v29_candidate'] = dict(name=name, parent_sha256=c['r0_sha256'], rule_sha256=rule_sha,
                                      reverted=sorted(set(c['candidates']['G16']['kept_consumers']) - set(kept)),
                                      kept_consumers=kept, reader_values_changed=False, training_steps=0)
        d = out_root / f'{cell}_{name}'
        d.mkdir()
        p = d / f'{task}.pt'
        with p.open('xb') as f:
            torch.save(child, f)
        manifest[cell][name] = dict(adapter_dir=str(d), sha256=sha(p), derived=True, physical_sources=n_src)
    print(cell, {k: v.get('physical_sources', 16) for k, v in manifest[cell].items()}, flush=True)
with (out_root / 'manifest.json').open('x') as f:
    json.dump(dict(rule_sha256=rule_sha, cells=manifest), f, indent=1)
print('BUILT', len(manifest))
