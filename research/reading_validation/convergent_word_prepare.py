"""Certify actual public English or Spanish word endpoints in a complete stimulus sentence."""
from collections import defaultdict
from pathlib import Path
from hashlib import sha256
import re,csv
from tokenizers import Tokenizer
from research.backbone_transfer.extract import digest,write_json,stream_rows
from research.reading_validation.identity_prepare import DATA,TOKENIZER,sealed_rows
from research.reading_validation.spanish_clue_data import ROOT as SPANISH

ROOT=DATA/'convergent';OUT=Path('research/results/reading-validation-convergent');PLAN=Path(__file__).with_name('convergent-word-protocol.md')


def run():
    OUT.mkdir(exist_ok=True);tok=Tokenizer.from_file(str(TOKENIZER));src=ROOT/'items.csv';groups=defaultdict(list);packet=[];maps=[]
    for r in csv.DictReader(src.open()):groups[r['sent_id']].append(r)
    def add(text,article,language,item_ids,expected=None):
        matches=list(re.finditer(r'\S+',text));enc=tok.encode(text,add_special_tokens=False);bounds=[]
        for m in matches:
            a,b=m.span();ix=[i for i,(l,u) in enumerate(enc.offsets) if l<b and u>a and text[max(a,l):min(u,b)].strip()]
            if not ix or enc.offsets[ix[-1]][1]!=b or tok.encode(text[:b],add_special_tokens=False).ids!=enc.ids[:ix[-1]+1]:raise ValueError('Printed word prefix or token boundary differs')
            bounds.append([a,b,ix[0],ix[-1]])
        if len(bounds)!=len(item_ids) or (expected and [m.group() for m in matches]!=expected):raise ValueError('Published original word order differs')
        k=sha256(text.encode()).hexdigest();packet.append(dict(id=k,text=text,words=bounds));maps.append(dict(id=k,article=article,language=language,items=item_ids))
    for sid,rows in sorted(groups.items(),key=lambda v:int(v[0])):
        rows=sorted(rows,key=lambda r:int(r['item_id']));sentence=rows[0]['sentence'];words=sentence.split();items=[None]*len(words)
        for row in rows:
            if row['sentence']!=sentence:raise ValueError('Unmatched original English cloze prefix')
            sentence+=' '+row['word'];items.append(int(row['item_id']));words.append(row['word'])
        add(sentence,sid,'English',items,words)
    original={r['id']:r for r in stream_rows(SPANISH/'mapping.jsonl')}
    for k,r in original.items():
        words=r['first']+r['middle']+r['last'];add(' '.join(words),k,'Spanish',list(range(1,22)),words)
    sealed_rows(ROOT/'stimuli.jsonl',packet);sealed_rows(ROOT/'map.jsonl',maps)
    write_json(OUT/'input-seal.json',{'plan':digest(PLAN),'source_items':digest(src),'english_endpoint_table':digest(ROOT/'outcomes.csv'),'stimuli':digest(ROOT/'stimuli.jsonl'),'map':digest(ROOT/'map.jsonl'),'documents':len(packet),'original_word_ends':sum(len(r['words']) for r in packet),'english_articles':len(groups),'spanish_articles':len(original)})
    print('Certified sources:',len(groups),'English,',len(original),'Spanish')

if __name__=='__main__':run()
