"""Request-scoped v25 controls. No checkpoint, selector or cold protocol changes."""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
import functools
import json
import threading
from .control import ControlStats, controlled_attention
from .admission import ReuseRegistry, admission_scope

# Existing runners and monkeypatches assume one request at a time per process.
_LOCK = threading.Lock()

@contextmanager
def install_controls(*, admission='original', repair_values='native', audit_path=None,
                     max_proxy_tokens=128):
    if admission not in ('original', 'reuse-prefetch') or repair_values not in ('native', 'source-proxy'):
        raise ValueError('Unsupported control mode')
    if (admission != 'original' or repair_values != 'native') and audit_path is None:
        raise ValueError('Changed paths require an audit file')
    from contiguous_fuxian.sharedkv_v25 import runtime as online
    from contiguous_fuxian.sharedkv.io import query_hash
    old_install = online.Runtime.install

    def installed(self, runner):
        old_install(self, runner)
        previous = runner.greedy_flexgen_completion
        old_variant = runner.runtime_variant
        runner.runtime_variant = lambda **kw: old_variant(**kw) + f'+v26-{admission}-{repair_values}'
        @functools.wraps(previous)
        def completion(*args, **kw):
            if not _LOCK.acquire(blocking=False):
                raise RuntimeError('v26 controls require one inference request per process')
            if repair_values != 'native' and self.mode != 'shared':
                _LOCK.release()
                raise ValueError('Native-patch intervention requires a shared checkpoint')
            registry = ReuseRegistry()
            stat = ControlStats()
            old_attention = online.repaired_attention
            context = admission_scope(kw['loader'], registry) if admission != 'original' else nullcontext()
            status, error = 'failed', None
            def changed(*pa, **kwargs):
                return controlled_attention(old_attention, *pa, repair_values=repair_values,
                                            stats=stat, max_proxy_tokens=max_proxy_tokens, **kwargs)
            try:
                if repair_values != 'native':
                    online.repaired_attention = changed
                with context:
                    result = previous(*args, **kw)
                if repair_values == 'source-proxy' and stat.consumer_calls == 0:
                    raise RuntimeError('No native patch was intervened on; result is not a valid control')
                status = 'complete'
                return result
            except BaseException as exc:
                error = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                online.repaired_attention = old_attention
                details = registry.finish()
                _LOCK.release()
                if audit_path:
                    row = {'query_hash': query_hash(kw['query_token_ids']),
                           'task': kw['loader']._info.task, 'admission': admission,
                           'repair_values': repair_values, 'status': status, 'error': error,
                           'counterfactual_quality_only': repair_values != 'native',
                           'proxy_control': asdict(stat), 'admission_stats': details,
                           'scope': 'Whole request incl. label scoring; not a hardware I/O log. '
                                    'No speed certification is implied by a completed request.'}
                    p = Path(audit_path); p.parent.mkdir(parents=True, exist_ok=True)
                    with p.open('a', encoding='utf8') as f:
                        f.write(json.dumps(row, allow_nan=False) + '\n')
        runner.greedy_flexgen_completion = completion
    online.Runtime.install = installed
    try:
        yield
    finally:
        online.Runtime.install = old_install
