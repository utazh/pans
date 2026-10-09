import json, glob, sys
B = '/home/shensicheng/qyd/pans/'
fs = sorted(glob.glob(B + 'src/studies/v29/a100_sys_q7/runs/*preserve*RAPS*b00*.v26.jsonl') or glob.glob(B + 'src/studies/v29/a100_sys_q7/runs/*.v26.jsonl'))
print(len(fs), fs[:2])
f = fs[0]
rows = [json.loads(l) for l in open(f)]
print('n rows', len(rows))
def walk(d, pre=''):
    for k, v in d.items():
        if isinstance(v, dict) and len(pre) < 40: walk(v, pre + k + '.')
        elif not isinstance(v, (list, dict)) or (isinstance(v, list) and len(v) < 5): print('  ', pre + k, '=', str(v)[:90])
walk(rows[0])
ex = [g for g in glob.glob(f.replace('.v26.jsonl', '*'))]
print(ex)
cl = f.replace('.v26.jsonl', '.v26.jsonl.effective_cli.json')
try: print(open(cl).read()[:3000])
except Exception as e: print(e)
