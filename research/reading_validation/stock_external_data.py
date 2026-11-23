"""Seal independent W00 article text apart from the human BIO-like categories."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import re
import zipfile
from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.definition_data import OUT,ROOT,word_geometry
from research.reading_validation.identity_prepare import TOKENIZER,sealed_rows


def prepare():
    tok=Tokenizer.from_file(str(TOKENIZER));known={re.sub(r'\W','',r['text'].casefold()) for r in stream_rows(ROOT/'stimuli.jsonl')}
    docs={};gold={};counts=Counter()
    with zipfile.ZipFile(ROOT/'w00.zip') as z:
        def rows(name):
            return z.read('W00_dataset/annotated.'+name).decode('utf-8').splitlines()
        for words,tags,meta in zip(rows('word'),rows('tag'),rows('meta'),strict=True):
            a=words.split();b=tags.split();m=meta.split(' $ ');counts['rows']+=1
            if len(a)!=len(b) or not a or len(m)!=5:
                counts['invalid_structure']+=1;continue
            text=' '.join(a);key=sha256(text.encode()).hexdigest()
            if re.sub(r'\W','',text.casefold()) in known:counts['deft_overlap']+=1;continue
            spans=[];left=0
            for w,t in zip(a,b,strict=True):
                right=left+len(w)
                if spans and spans[-1][0]==t:spans[-1][2]=right
                else:spans.append([t,left,right])
                left=right+1
            terms=[s[1:] for s in spans if s[0]=='TERM'];definitions=[s[1:] for s in spans if s[0]=='DEF']
            if m[0]=='0' and not terms and not definitions:kind='negative'
            elif m[0]=='1' and len(terms)==len(definitions)==1 and terms[0][1]<=definitions[0][0]:kind='single_forward'
            else:kind='ambiguous_or_reverse'
            ws,n=word_geometry(text,tok)
            if not ws or n>1024:counts['unsupported_doc']+=1;continue
            if key in docs:
                counts['duplicate']+=1
                previous=gold[key]
                if previous['terms']!=terms or previous['definitions']!=definitions:
                    previous['kind']='conflicting_duplicate'
                continue
            docs[key]={'id':key,'text':text,'words':ws};gold[key]={'id':key,'source_section':m[1],'kind':kind,'terms':terms,'definitions':definitions}
    p=ROOT/'w00-stimuli.jsonl';y=ROOT/'w00-structural.jsonl';sealed_rows(p,[docs[k] for k in sorted(docs)]);sealed_rows(y,[gold[k] for k in sorted(gold)])
    write_json(OUT/'stock-external-lock.json',{'protocol':digest(Path(__file__).with_name('stock-probability-protocol.md')),'archive':digest(ROOT/'w00.zip'),
        'stimuli':digest(p),'grades':digest(y),'counts':dict(counts),'scope':dict(Counter(v['kind'] for v in gold.values())),'rows_encoded':len(docs)})


if __name__=='__main__':prepare()
