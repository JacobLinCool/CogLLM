"""Locate source-specific explanations and genuinely competing open symbols in scientific prose."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
import re

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.identity_prepare import DATA, sealed_rows

ROOT=DATA/'symbols'
OUT=Path('research/results/reading-validation-symbols')


def connected(keys, edges):
    parent={k:k for k in keys}
    def find(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]];k=parent[k]
        return k
    for a,b in edges:
        if a in parent and b in parent:parent[find(a)]=find(b)
    return {k:find(k) for k in keys}


def source_links(row):
    entities=row['entity'];text=row['text'];rels=list(row['relation'].values())
    symbols={k:(v['start'],v['end']) for k,v in entities.items() if v['label']=='SYMBOL'}
    if any(text[v['start']:v['end']]!=v['text'] for v in entities.values()):return None
    components=connected(symbols,[(v['arg0'],v['arg1']) for v in rels if v['label']=='Corefer-Symbol'])
    same=defaultdict(set);definitions=defaultdict(set)
    for k,span in symbols.items():same[components[k]].add(span)
    for relation in rels:
        if relation['label']!='Direct':continue
        a,b=relation['arg0'],relation['arg1']
        if a in symbols and b not in symbols:a,b=b,a
        if b not in symbols or a not in entities or entities[a]['label']!='PRIMARY':continue
        v=entities[a];definitions[components[b]].add((v['start'],v['end']))
    part=[]
    for k,terms in same.items():
        spans=sorted(terms);regions=sorted(definitions[k]);opening=spans[0]
        if any(a<opening[1] for a,b in regions):
            state='already_or_concurrently_explained'
        elif regions:
            state='forward'
        else:state='no_direct_explanation_in_paragraph'
        part.append({'sources':[list(s) for s in spans],'definitions':[list(s) for s in regions],'kind':state})
    return sorted(part,key=lambda r:r['sources'][0])


def competing(issues):
    eligible=[q for q in issues if q['kind']=='forward']
    result=[]
    for i,a in enumerate(eligible):
        for b in eligible[i+1:]:
            a0,a1=a['sources'][0];b0,b1=b['sources'][0];a2,a3=a['definitions'][0];b2,b3=b['definitions'][0]
            common=max(a1,b1)
            if common<=min(a2,b2) and (a3<=b2 or b3<=a2):result.append((a,b))
    return result


def scan_symlink():
    rows=[];meta=defaultdict(Counter);seen=set();files={}
    for phase in ('train','dev'):
        for path in sorted((ROOT/'symlink'/phase).glob('*.json')):
            files[str(path.relative_to(ROOT))]=digest(path)
            for k,q in read_json(path).items():
                if k in seen:continue
                seen.add(k);issues=source_links(q)
                if issues is None:
                    meta[phase]['unmatched_source_offsets']+=1;continue
                meta[phase]['paragraphs']+=1;pure=q['document'].partition('v')[0]
                data={'id':sha256(q['text'].encode()).hexdigest(),'paper':pure,'source':k,'text':q['text'],'issues':issues,'published':phase}
                pairs=competing(issues);meta[phase]['forward_sources']+=sum(x['kind']=='forward' for x in issues)
                meta[phase]['pure_dual_source_cases']+=len(pairs);meta[phase]['documents_with_competition']+=bool(pairs)
                rows.append(data)
    return rows,{k:dict(v) for k,v in meta.items()},files


if __name__=='__main__':
    rows,meta,files=scan_symlink();print(json.dumps(meta,indent=2))
