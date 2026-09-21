"""Generate predeclared one-hop source-budget stages; never choose on accuracy."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .records import load,FIELDS,PLAN_FORMAT,json_digest,validate_sources,assert_plan


def proposals(rows,block_size=4,protect_first=4):
    header=next(iter(rows.values()));n=header['geometry']['layers'];period=header['period']
    if block_size<2 or period%block_size or protect_first<0 or protect_first>n:
        raise ValueError('Block size must divide period; protection must be in layer range')
    out=[]
    for start in range(0,n,block_size):
        end=min(start+block_size,n)
        candidates=[i for i in range(start,end) if i>=protect_first and i%period]
        # The first member of a four-layer block is initially native. This also
        # avoids hiding the observed 77/78-block budget issue via a union.
        candidates=[i for i in candidates if i!=start]
        groups={}
        for lid in candidates:
            signature=tuple(tuple(r['selection_by_layer'][lid]) for _,r in sorted(rows.items()))
            groups.setdefault(signature,[]).append(lid)
        classes=sorted(groups.values(),key=lambda x:(-len(x),x))
        if not classes or len(classes[0])<2:
            out.append({'start':start,'end':end,'status':'no_compatible_group','source':None,'consumers':[]});continue
        members=classes[0]
        source=members[len(members)//2]  # structural, no quality-based search
        out.append({'start':start,'end':end,'status':'structural_candidate',
                    'source':source,'consumers':[i for i in members if i!=source],
                    'native':[i for i in range(start,end) if i==source or i not in members]})
    return out


def build_stages(rows,block_size=4,protect_first=4):
    header=next(iter(rows.values()));n=header['geometry']['layers']
    candidates=proposals(rows,block_size,protect_first)
    sources=list(range(n));stages=[]
    for group in reversed(candidates):
        if not group['consumers']:continue
        for lid in group['consumers']:sources[lid]=group['source']
        validate_sources(sources,header['period'])
        for row in rows.values():assert_plan(row,sources)
        stages.append({'format':PLAN_FORMAT,**{k:header[k] for k in FIELDS},
                       'sources':sources.copy(),'unique_sources':len(set(sources)),
                       'added_group':group,'design_query_hashes':sorted(rows),
                       'status':'structural_research_plan_not_quality_accepted',
                       'policy':{'block_size':block_size,'protect_first':protect_first,
                                 'order':'late_to_early','source_rule':'median_of_largest_compatible_nonleader_class'},
                       'warning':'No task quality or speed forecast. Each stage needs independent dev evaluation.'})
    return candidates,stages


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-fit',required=True);p.add_argument('--task',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--protect-first',type=int,default=4)
    p.add_argument('--block-size',type=int,default=4)
    a=p.parse_args();rows=load(a.teacher_fit,a.task,'fit');out=Path(a.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    candidates,stages=build_stages(rows,a.block_size,a.protect_first)
    (out/'candidates.json').write_text(json.dumps(candidates,indent=2))
    for stage in stages:
        path=out/f"g{stage['unique_sources']:02d}.json"
        path.write_text(json.dumps(stage,indent=2));print(path)
    if not stages:raise ValueError('No compatible research topology; inspect candidates.json')

if __name__=='__main__':main()
