"""Seal authentic physical-state, passage-coherence and first-response materials separately."""
import csv
from collections import Counter,defaultdict
import json
from pathlib import Path

from research.reading_validation.memory_data import BASE,h,load_rows,sha

TRIP=BASE/'world-states'
ITALIAN=BASE/'italian-order'
OUT=Path('research/results/reading-validation-state-transport')
PLAN=Path(__file__).with_name('state-transport-protocol.md')


def italian(split):
    with (ITALIAN/(split+'.tsv')).open(encoding='utf8',newline='') as f:
        yield from csv.DictReader(f,delimiter='\t')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    original=json.loads((TRIP/'www.json').read_bytes());articles={};details={};design=defaultdict(dict)
    def add(t):
        k=h(t)
        if articles.setdefault(k,t)!=t:raise ArithmeticError('Nonunique article digest')
        return k
    def join(ss):return '\n'.join(ss)
    families={}
    for split,rr in original.items():
        bank={}
        for name,r in rr.items():
            if r['type'] not in (None,'cloze') or r['aug']:continue
            g=r['story_id']
            if families.setdefault(g,split)!=split:raise ValueError('Physical story family occurs on both sides of a split')
            text=join(r['sentences']);key=add(text)
            if r['length']!=len(r['sentences']):raise ValueError('Missing physical story sentence')
            base=rr[str(g)];changes=[i for i,(a,b) in enumerate(zip(r['sentences'],base['sentences'],strict=True)) if a!=b]
            if (r['type'] is None and changes) or (r['type']=='cloze' and len(changes)!=1):
                raise ValueError(f'Unverified single-sentence intervention {split}, {name}, {changes}')
            end=0;spans=[]
            for s in r['sentences']:
                a=end;b=a+len(s);spans.append([a,b]);end=b+1
            bank[name]=dict(key=key,family=g,id=name,spans=spans,original=add(join(base['sentences'])),change=changes)
        design['physical'][split]=bank
    for split in ('train','held'):
        design['italian'][split]=[{ 'key':add(r['TEXT']), 'id':r['ID']} for r in italian(split)]
    original_key={}
    for r in load_rows(BASE/'memory/stories.jsonl'):
        text='\n\n'.join(r['sentences']);original_key[r['id']]=add(text)
    design['frick']=original_key
    smeta=json.loads(Path('research/results/reading-validation-transition-external/stimuli.json').read_bytes())
    eligible={r['doc'] for r in smeta['material']}
    design['english_coherence']={}
    for r in load_rows(BASE/'cohesion-stimuli.jsonl'):
        if r['id'] in eligible:
            design['english_coherence'][r['id']]=add(r['text'])
    files={'world-states/www.json':sha(TRIP/'www.json'),'italian-order/train.tsv':sha(ITALIAN/'train.tsv'),'italian-order/held.tsv':sha(ITALIAN/'held.tsv')}
    with (TRIP/'stimuli.jsonl').open('w') as f:
        for k,t in sorted(articles.items()):f.write(json.dumps(dict(id=k,text=t),ensure_ascii=False)+'\n')
    if set(design['english_coherence'])!=eligible:raise ValueError('English coherence material extraction incomplete')
    (OUT/'inputs.json').write_text(json.dumps(dict(protocol=sha(PLAN),raw=files,groups=design,source=sha(TRIP/'stimuli.jsonl')),separators=(',',':'))+'\n')
    print('Articles',len(articles),'TRIP',[(s,len(v)) for s,v in design['physical'].items()],'Italian',[(s,len(v)) for s,v in design['italian'].items()],'English',len(eligible),'Frick',len(original_key))


if __name__=='__main__':prepare()
