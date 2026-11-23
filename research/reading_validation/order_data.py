"""Certify paired source clauses and order manipulations without reading behavior as labels."""

from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import re

from research.reading_validation.memory_data import BASE,h,rows_xlsx,sha

WORK=BASE/'discourse-order'
OUT=Path('research/results/reading-validation-discourse-order')
PLAN=Path(__file__).with_name('order-protocol.md')
FILES={'materials.xlsx':'03c3b9f4e0e7f5391354032e03652af3e31118536917e7cde497f10a208d3d60',
       'full.tsv':'b61b483c6fe01e7595b1fe7fcf640c252a14cbc4936d064285f2f9e31c4d2ffb',
       'regions.tsv':'5cb1892a00069c13d8e46eedc965fd38234ec8c7d6f2bb7567fcfc96647ddd48'}


def table(file):
    with (WORK/file).open() as f:
        yield from csv.DictReader(f,delimiter='\t')


def group_split(groups,salt):
    order=sorted(set(groups),key=lambda k:h('cogllm-order-20261122.'+salt+'.'+k))
    return {k:'train' if i<int(.6*len(order)) else 'dev' if i<int(.75*len(order)) else 'test' for i,k in enumerate(order)}


def audit():
    if any(sha(WORK/k)!=v for k,v in FILES.items()):
        raise ValueError('Original OSF file differs from its author-supplied checksum')
    labels=defaultdict(set);readers=set()
    for r in table('regions.tsv'):
        labels[r['itemid'],r['condition']].add(r['IA_LABEL']);readers.add(r['participant'])
    groups=defaultdict(list);invalid=defaultdict(list)
    for r in rows_xlsx(WORK/'materials.xlsx',{'itemID','condition','event order','clause order','givenness','sentence'},require='sentence'):
        article=r['sentence'];item,condition=r['itemID'],r['condition']
        unique=labels[item,condition]
        if len(unique)!=1:
            raise ValueError('Missing or conflicting source-defined second-clause spans')
        original=next(iter(unique));pattern=r'\s+'.join(map(re.escape,original.split()))
        matches=list(re.finditer(pattern,article,re.IGNORECASE))
        if len(matches)!=1:
            invalid[item].append([condition,'second_clause_not_unique_in_original']);continue
        b,e=matches[0].span()
        previous=list(re.finditer(r'[.!?][\'”"]?\s+',article[:b]))
        if not previous:
            invalid[item].append([condition,'unidentified_sentence_beginning']);continue
        a=previous[-1].end()
        connectives=list(re.finditer(r'\b(?:before|after)\b',article[a:e],re.IGNORECASE))
        if (len(connectives)!=1 or (r['clause order']=='s-m')!=(connectives[0].start()==0)
            or article[e-1] not in '.!?'):
            invalid[item].append([condition,'connective_or_sentence_mismatch']);continue
        groups[item].append(dict(key=h(article),condition=condition,article=article,target=[a,e],
                      second=[b,e],connective=[a+connectives[0].start(),a+connectives[0].end()],
                      given=r['givenness'],order=r['event order'],structure=r['clause order']))
    missing=set(i for i,_ in labels)-set(groups)
    if any(len(groups[g])+len(invalid.get(g,[]))!=8 for g in set(groups)|missing):
        raise ValueError('Not every original experimental arm was accounted for')
    valid={g:v for g,v in groups.items() if len(v)==8}
    material=group_split(valid,'material');reader=group_split(readers,'reader')
    texts={r['key']:r['article'] for arms in valid.values() for r in arms}
    if any(len({(r['given'],r['order'],r['structure']) for r in v})!=8 for v in valid.values()):
        raise ValueError('Missing factorial cell in an eligible original family')
    for group in valid.values():
        for r in group:r.pop('article')
    with (WORK/'stimuli.jsonl').open('w') as f:
        for k,v in sorted(texts.items()):
            f.write(json.dumps(dict(id=k,text=v),ensure_ascii=False)+'\n')
    OUT.mkdir(parents=True,exist_ok=True)
    meta=dict(registration=sha(PLAN),original=FILES,inventory=len(labels),eligible=valid,excluded=dict(invalid),
              material_split=material,reader_split={h(k):v for k,v in reader.items()},text=sha(WORK/'stimuli.jsonl'))
    (OUT/'inputs.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    print('Eligible matched passage families',len(valid),'from',len(set(i for i,_ in labels)),'trials sources',len(readers),'unique text',len(texts))


if __name__=='__main__':audit()
