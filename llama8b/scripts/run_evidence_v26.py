#!/usr/bin/env python3
"""v26 controls -- v25 model options -- v23 options -- v21 options -- round4 options."""
from pathlib import Path
import argparse, json, runpy, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

def value(argv, flag, default=None):
    if argv.count(flag) > 1: raise ValueError(f'Duplicate flag {flag}')
    if flag not in argv: return default
    i = argv.index(flag)
    if i + 1 == len(argv): raise ValueError(f'Missing {flag} value')
    return argv[i + 1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--admission', choices=['original','reuse-prefetch'], default='original')
    p.add_argument('--repair-values', choices=['native','source-proxy'], default='native')
    p.add_argument('--audit-jsonl', type=Path)
    p.add_argument('--max-proxy-tokens', type=int, default=128)
    p.add_argument('rest', nargs=argparse.REMAINDER)
    a = p.parse_args()
    args = a.rest[1:] if a.rest[:1] == ['--'] else a.rest
    if args.count('--') < 3:
        raise ValueError('Need v25 options -- v23 options -- v21 options -- round4 options')
    if a.audit_jsonl and a.audit_jsonl.exists(): raise FileExistsError(a.audit_jsonl)
    if a.admission != 'original' and value(args,'--mode') == 'original':
        raise ValueError('Use identity as matched executor baseline for admission controls')
    if a.repair_values != 'native':
        if '--diagnostic' not in args or value(args,'--mode') != 'shared':
            raise ValueError('source-proxy requires --diagnostic and --mode shared')
        if value(args,'--cold','none') != 'none' or '--nvtx' in args:
            raise ValueError('source-proxy is a quality intervention, not a cold/latency benchmark')
    from contiguous_fuxian.sharedkv_v26.guard import check
    from contiguous_fuxian.sharedkv_v26.integration import install_controls
    check(ROOT)
    # Log the final serving CLI after the model/store/cache override wrappers.
    # The inherited round4 command.json retains its pre-override template.
    import contiguous_fuxian.flexgen_qwen_reprefill as runner
    old_main = runner.main
    def effective_main(*pa, **kw):
        if a.audit_jsonl:
            a.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with Path(str(a.audit_jsonl)+'.effective_cli.json').open('x') as f:
                json.dump({'argv':sys.argv,'admission':a.admission,'repair_values':a.repair_values},f,indent=2)
        return old_main(*pa, **kw)
    runner.main = effective_main
    old = sys.argv
    try:
        with install_controls(admission=a.admission,repair_values=a.repair_values,
                              audit_path=a.audit_jsonl,max_proxy_tokens=a.max_proxy_tokens):
            sys.argv = [str(ROOT/'scripts/run_native_prefix_v25.py'), *args]
            runpy.run_path(str(ROOT/'scripts/run_native_prefix_v25.py'), run_name='__main__')
    finally:
        sys.argv = old
        runner.main = old_main

if __name__ == '__main__': main()
