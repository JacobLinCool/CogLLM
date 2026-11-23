"""Recover original standalone arguments, declared interventions and blinded inference material."""
from collections import Counter,defaultdict
import csv
from io import StringIO
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from research.reading_validation.argument_data import chunks
from research.reading_validation.argument_labels import WORK,fingerprint
from research.reading_validation.memory_data import BASE,h,sha,load_rows
from research.reading_validation.argument_data import OUT as ESSAYS

DATA=BASE/'reason-quality'
OUT=Path('research/results/reading-validation-reason-quality')
PLAN=Path(__file__).with_name('reason-quality-protocol.md')
DIMENSIONS=('LocalSufficiency','Arrangement','Clarity')

def experts(scores=False,ids=None):
    with ZipFile(DATA/'expert-v2.zip') as z:
        for name in sorted(z.namelist()):
            if name.startswith('__MAC') or not name.endswith('.xmi'):continue
            topic,side,origin=name.split('/')[-3:]
            if ids is not None and origin not in ids:continue
            root=ET.fromstring(z.read(name));fields={p.tag.split('}')[-1]:p for p in root}
            t=fields['Sofa'].attrib['sofaString']
            row=dict(text=t,topic=topic,side=side,record=origin)
            if scores:
                vals={}
                for label in DIMENSIONS:
                    f=fields[label].attrib
                    raw=[float(x) for x in f['allScores'].split()]
                    if len(raw)!=3 or any(x not in (1.,2.,3.) for x in raw) or abs(sum(raw)/3-float(f['mean']))>.001:
                        raise ValueError('Expert votes differ from author-reported judgment')
                    vals[label]=list(map(int,raw))
                row['votes']=vals
            yield row

def webis():
    with (DATA/'webis-text.csv').open(newline='') as f:yield from csv.DictReader(f)

def ukp(scores=False):
    meta=json.loads((ESSAYS/'inputs.json').read_bytes());by_id={v['id']:k for k,v in meta['articles'].items() if v['role']=='external'}
    corpus={r['id']:r['text'] for r in load_rows(WORK/'stimuli.jsonl')}
    with ZipFile(WORK/'votes.zip') as z:
        votes={(int(r['ESSAY']),int(r['ARGUMENT'])):r for r in csv.DictReader(StringIO(z.read('Annotations/annotations.csv').decode(),newline=''),delimiter=';')}
    with ZipFile(WORK/'sufficiency.zip') as z:
        for r in csv.DictReader(StringIO(z.read('UKP-InsufficientArguments_v1.0/data-tokenized.tsv').decode('latin1'),newline=''),delimiter='\t'):
            fam,num=int(r['ESSAY']),int(r['ARGUMENT']);k=by_id.get(fam)
            if k is None:continue
            orig=[m.group() for m in re.finditer(r'[^\r\n]+',corpus[k]) if fingerprint(m.group())==fingerprint(r['TEXT'])]
            if len(orig)!=1:continue
            observed=votes.get((fam,num));t=observed['TEXT'] if observed else orig[0]
            if fingerprint(t)!=fingerprint(orig[0]):raise ValueError('Source rater received a different argument')
            row=dict(text=t,family=fam,record=num,direct=observed is not None)
            if scores:
                row['votes']=[int(observed[idn]!='Sufficiency') for idn in ('A1','A2','A3')] if observed else [int(r['ANNOTATION']!='insufficient')]
            yield row

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);texts={};registry={}
    def register(t):
        k=h(t)
        if not t.strip() or texts.setdefault(k,t)!=t:raise ValueError('Empty or nonunique original argument')
        return k
    ex=[{**{c:r[c] for c in ('topic','side','record')},'key':register(r['text'])} for r in experts()]
    dd=[{**{c:r[c] for c in ('family','record','direct')},'key':register(r['text'])} for r in ukp()]
    source={};w=[]
    for r in webis():
        if r['Is Argument?']!='True':continue
        a=(r['Discussion ID'],r['Argument ID']);k=register(r['Premise'])
        if a in source and source[a]!=k:raise ValueError('Ambiguous source debate argument')
        source[a]=k;w.append(dict(key=k,topic=r['Topic ID'],id=list(a)))
    # Prevent verbatim or 20-word argument reuse from either source in the final expert panel.
    direct=set(r['key'] for r in dd+w);windows=set().union(*(chunks(texts[k]) for k in direct));expert_keys={r['key'] for r in ex}
    overlap={k for k in expert_keys if k in direct or bool(chunks(texts[k])&windows)}
    count=Counter(r['key'] for r in ex);repeated={k for k,v in count.items() if v>1}
    for r in ex:r['eligible']=r['key'] not in overlap|repeated
    with (DATA/'stimuli.jsonl').open('w') as f:
        for k,t in sorted(texts.items()):f.write(json.dumps(dict(id=k,text=t),ensure_ascii=False)+'\n')
    registry=dict(protocol=sha(PLAN),source=sha(DATA/'stimuli.jsonl'),expert=ex,webis=w,ukp=dd,
           overlap=len(overlap),repeated_expert=len(repeated),raw={f:sha(DATA/f) for f in ('expert-v2.zip','webis-text.csv','webis-choice.csv')})
    (OUT/'material.json').write_text(json.dumps(registry,separators=(',',':'))+'\n')
    print('Original arguments',len(texts),'source sufficiency',len(dd),'logical arguments',len(source),'held expert arguments',len(ex),'overlap',len(overlap),'duplicates',len(repeated))

if __name__=='__main__':prepare()
