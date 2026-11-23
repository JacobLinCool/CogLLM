"""Recover both exact randomized word-by-word presentations from the authors' experiment."""
from collections import defaultdict
import csv
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from zipfile import ZipFile
from tokenizers import Tokenizer
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.identity_prepare import DATA,TOKENIZER,sealed_rows

ROOT=DATA/'multilab'
OUT=Path('research/results/reading-validation-multilab')
PLAN=Path(__file__).with_name('multilab-word-protocol.md')


def trial_metadata(source):
    values=defaultdict(set)
    with source.open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            values[int(r['item'])].add(float(r['cloze']))
    return values


def build():
    zpath=ROOT/'stimuli.zip'
    with ZipFile(zpath) as outer,ZipFile(BytesIO(outer.read('Stimuli/Presentation/Delong_Presentation_updated.zip'))) as inner:
        experiment=inner.read('Delong/1_delong_presentation/Delong.pcl')
    code=experiment.decode('cp1252')
    raw=code.split('array <string> stimuli_Delong[nstimuli_Delong][16] = {',1)[1].split('};',1)[0]
    pairs=[re.findall(r'"([^"]*)"',line) for line in raw.splitlines() if line.strip()]
    if len(pairs)!=80 or any(len(pair)!=16 for pair in pairs):raise ValueError('Unexpected authentic presentation table')
    tok=Tokenizer.from_file(str(TOKENIZER));texts=[];targets=[];unmatched=[]
    metadata={k:trial_metadata(ROOT/(k+'.txt')) for k in ('article','noun')}
    for item,pair in enumerate(pairs,101):
        anchors=[]
        for base in (0,8):
            text,index,article,condition,article_norm,noun_norm=pair[base:base+6];i=int(index)
            words=list(re.finditer(r'\S+',text));parts=[m[0] for m in words]
            if parts[i]!=article or article not in ('a','an') or condition not in ('expected','unexpected'):raise ValueError('Article differs from actual signal marker')
            enc=tok.encode(text,add_special_tokens=False);edges=[]
            for m in words:
                a,b=m.span();native=[j for j,(l,h) in enumerate(enc.offsets) if l<b and h>a and text[max(l,a):min(h,b)].strip()]
                if not native or enc.offsets[native[-1]][1]!=b or tok.encode(text[:b],add_special_tokens=False).ids!=enc.ids[:native[-1]+1]:raise ValueError('Printed word is not prefix-certified')
                edges.append([a,b,native[0],native[-1]])
            key=sha256(text.encode()).hexdigest();texts.append(dict(id=key,text=text,words=edges));anchors.append(parts[:i])
            row=dict(id=key,item=item,condition=condition,article=i,noun=i+1)
            for field,val in [('article',article_norm),('noun',noun_norm)]:
                # The experiment reserves port code 200 for a zero cloze value.
                cloze=0 if int(val)==200 else int(val)
                row[field+'_cloze']=cloze
                if cloze not in metadata[field][item]:unmatched.append([item,condition,field,cloze,sorted(metadata[field][item])])
            targets.append(row)
        if anchors[0]!=anchors[1]:raise ValueError('The two printed passages differ before their critical article')
    sealed_rows(ROOT/'stimuli.jsonl',texts);sealed_rows(ROOT/'conditions.jsonl',targets)
    write_json(OUT/'input-seal.json',dict(protocol=digest(PLAN),registration=digest(OUT/'registration.json'),source_materials=digest(zpath),presentation=sha256(experiment).hexdigest(),tokenizer=digest(TOKENIZER),stimuli=digest(ROOT/'stimuli.jsonl'),conditions=digest(ROOT/'conditions.jsonl'),source_n400={k:digest(ROOT/(k+'.txt')) for k in metadata},items=len(pairs),texts=len(texts),words=sum(len(x['words']) for x in texts),unmatched_norm_codes=unmatched))
    print('Verified original exposures:',len(texts),'EEG norm metadata exceptions:',len(unmatched)); print(unmatched[:6])


if __name__=='__main__':build()
