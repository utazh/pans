import json
D = '/home/shensicheng/qyd/pans/src/studies/v29/a100_sys_q7/runs/'
for f in ['rte_b25_preserve_G24_b00/requests.progress.jsonl', 'rte_b25_preserve_G24_b00.cold.jsonl', 'rte_b25_preserve_G24_b00/scored_records.jsonl']:
    r = json.loads(open(D + f).readline())
    ks = [(k, v) for k, v in r.items() if not isinstance(v, (list, dict))]
    print('==', f, len(r), 'keys')
    print([k for k, v in ks if any(s in k for s in ('time', 'unix', 'wall', 'start', 'end', 'ms', 'epoch', 'stamp', 'perf', '_at'))])
    for k, v in r.items():
        if isinstance(v, (int, float)) and v > 1.6e9: print('  epoch-like', k, v)
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, (int, float)) and v2 > 1.6e9: print('  epoch-like', k + '.' + k2, v2)
