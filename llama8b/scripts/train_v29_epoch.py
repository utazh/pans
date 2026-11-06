"""Rule rev 2 A2 trainer wrapper: epoch-shuffled sampling WITHOUT replacement.
Replaces random.Random inside sharedkv_v24.train (its only use: rng.randrange for the fit
query of each step) with a seeded per-epoch permutation, logs every sampled index, then runs
the unchanged scripts/train_instrumented_v25.py.
Usage: train_v29_epoch.py --sample-log PATH <train_instrumented_v25 args...>"""
from pathlib import Path
import json, random, runpy, sys, types
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / 'src'))
i = sys.argv.index('--sample-log'); log = Path(sys.argv[i + 1]); del sys.argv[i:i + 2]

class EpochRandom:
    def __init__(self, seed):
        self.seed, self.epoch, self.queue, self.n, self.drawn = seed, 0, [], None, []
    def randrange(self, n):
        if self.n is None: self.n = n
        if n != self.n: raise ValueError('population size changed during training')
        if not self.queue:
            perm = list(range(n)); random.Random(f'{self.seed}:{self.epoch}').shuffle(perm)
            self.queue, self.epoch = perm, self.epoch + 1
        k = self.queue.pop(0); self.drawn.append(k)
        return k

holder = []
def factory(seed):
    rng = EpochRandom(seed); holder.append(rng); return rng

import contiguous_fuxian.sharedkv_v24.train as t
t.random = types.SimpleNamespace(Random=factory)
try:
    sys.argv = [str(R / 'scripts/train_instrumented_v25.py')] + sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    rng = holder[0] if holder else None
    counts = {}
    for k in (rng.drawn if rng else []): counts[k] = counts.get(k, 0) + 1
    with log.open('x') as f:
        json.dump(dict(population=rng.n if rng else None, steps=len(rng.drawn) if rng else 0,
                       epochs_started=rng.epoch if rng else 0, distinct=len(counts),
                       min_count=min(counts.values()) if counts else 0,
                       max_count=max(counts.values()) if counts else 0, drawn=rng.drawn if rng else []), f)
