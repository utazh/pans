from pathlib import Path
import json,statistics
r=Path(__file__).resolve().parents[1];out={}
for a in ['identity','joint']:
 logs=[]
 for b in range(10):
  p=r/f'results/v23_cold_b{b:02}_{a}.cold.jsonl'
  if p.exists():logs.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
 valid=[x for x in logs if x['status']=='completed']
 out[a]={'completed':len(valid),'all_zero_pages':all(x['page_after']['resident']==0 for x in valid),'failed':[x for x in logs if x['status']!='completed']}
 if valid:out[a]['last_ms']=valid[-1]['logits_ready_ms']
paired=[]
for b in range(10):
 ps=[r/f'results/v23_cold_b{b:02}_{a}/summary.json' for a in ['identity','joint']]
 if all(p.exists() for p in ps):
  vv=[json.loads(p.read_text())['overall'] for p in ps]
  paired.append({'block':b,'identity_ms':vv[0]['mean_ttft_ms'],'joint_ms':vv[1]['mean_ttft_ms'],'identity_accuracy':vv[0]['accuracy'],'joint_accuracy':vv[1]['accuracy']})
out['paired_cold_blocks']=paired
for name in ['main_resume','postprocess_resume','finish_resume']:
 p=r/f'logs/v23_{name}.log'
 if p.exists():
  ls=p.read_text().splitlines();out[name]={'tail':ls[-2:],'traceback':any('Traceback (most recent call last)' in s for s in ls)}
print(json.dumps(out,ensure_ascii=False))
