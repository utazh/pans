import json,html,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('profile_dir',type=Path);a=p.parse_args();r=a.profile_dir.resolve()
d=json.loads((r/'ttft_timeline_data.json').read_text());a,b=d['window_ns'];wall=(b-a)/1e6
colors={'selector':'#4169B1','decision':'#8158A3','wait':'#DFA039','cache':'#946A4B','decoder':'#148B82','mask':'#738399','other':'#CED5DF','index':'#9866B0','payload':'#DF7838'}
def cc(x):
 if 'wait_and_gather' in x or 'payload_resolve' in x:return colors['wait']
 if x in ['score_preparation','selection_decision']:return colors['decision']
 if 'selector' in x:return colors['selector']
 if 'decoder' in x:return colors['decoder']
 if 'cache_score' in x:return colors['cache']
 if 'mask' in x:return colors['mask']
 return colors['other']
gc=lambda x:colors.get({'selector_index':'index','selector_score':'selector','payload_or_other':'payload'}.get(x,x),colors['other'])
lanes=[('Main CPU',[(l,h,cc(c),c) for l,h,c in d['cpu_timeline']])]
for tid in sorted(set(e['tid'] for e in d['worker_nvtx'])):
 es=[(e['start'],e['end'],colors['index'],e['name']) for e in d['worker_nvtx'] if e['tid']==tid and e['name']=='_load_selector_keys_sync']
 if es:lanes.append(('Selector worker',es))
for label,title in [('payload/prepare_and_pack','KV prepare / pack'),('payload/host_gather','Native host gather')]:
 es=[(e['start'],e['end'],colors['payload'],e['name']) for e in d['worker_nvtx'] if e['name']==label]
 if es:lanes.append((title,es))
for stream in sorted(set(e['streamId'] for e in d['gpu_events'])):
 for kind in ['kernel','memcpy']:
  es=[(e['start'],e['end'],gc(e['category']),e['category']+'/'+kind) for e in d['gpu_events'] if e['streamId']==stream and e['kind']==kind]
  if es:lanes.append((f'GPU stream {stream}: {kind}',es))
panel_h=32*len(lanes)+110
height=105+2*panel_h+65
legend_y=height-45
out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{height}" viewBox="0 0 1400 {height}"><rect width="1400" height="{height}" fill="#fff"/><g font-family="Arial, sans-serif" fill="#172534">']
def text(x,y,t,size=16):out.append(f'<text x="{x}" y="{y}" font-size="{size}">{html.escape(t)}</text>')
text(30,38,'TTFT timeline: CPU ranges and CUDA activity',25)
text(30,67,f'Fixed P8 / {int(r.parent.name[1:])}% blocks / {r.name.upper()} diagnostic request 1. Separate from formal results.',15)
for top,limit,title in [(105,wall,f'Full logits-ready interval: {wall:.1f} ms'),(105+panel_h,min(160.,wall),f'Zoom: first {min(160.,wall):.0f} ms')]:
 text(30,top,title,19);left=245;w=1110;rowh=32
 for n in range(9):
  t=limit*n/8;x=left+w*n/8
  out.append(f'<path d="M{x},{top+15} V{top+25+rowh*len(lanes)}" stroke="#e8ecf0"/>')
  text(x-10,top+47+rowh*len(lanes),f'{t:.0f}',12)
 for i,(name,es) in enumerate(lanes):
  y=top+25+i*rowh;text(30,y+17,name,14)
  out.append(f'<rect x="{left}" y="{y}" width="{w}" height="23" fill="#f5f7fa"/>')
  for l,h,col,label in es:
   l=max(0,(l-a)/1e6);h=min(limit,(h-a)/1e6)
   if h<=l or l>=limit:continue
   x=left+l/limit*w;ww=(h-l)/limit*w
   out.append(f'<rect x="{x:.3f}" y="{y}" width="{ww:.3f}" height="23" fill="{col}"><title>{html.escape(label)}: {l:.3f} to {h:.3f} ms</title></rect>')
 text(left,top+69+rowh*len(lanes),'Elapsed time (ms)',13)
legend=[('Selector',colors['selector']),('Decision / index',colors['decision']),('Payload wait / gather',colors['wait']),('Cache scores',colors['cache']),('Decoder',colors['decoder']),('Mask',colors['mask']),('Other',colors['other'])]
x=30
for name,col in legend:
 out.append(f'<rect x="{x}" y="{legend_y}" width="16" height="16" fill="{col}"/>');text(x+23,legend_y+14,name,14);x+=len(name)*8+58
out.append('</g></svg>')
(r/'ttft_timeline.svg').write_text(''.join(out))
print('SVG timeline written',len(lanes),'lanes')
