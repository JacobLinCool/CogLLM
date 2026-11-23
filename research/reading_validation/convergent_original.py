"""Preserve the exact Germanic apostrophe and punctuation exposures in the English EEG experiment."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import csv
import re
from tokenizers import Tokenizer
from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.convergent_word_prepare import OUT, ROOT
from research.reading_validation.identity_prepare import TOKENIZER,sealed_rows


def run():
    groups=defaultdict(list)
    for row in csv.DictReader((ROOT/'outcomes.csv').open()):
        groups[row['sent_id']].append((int(row['item_id']),row['sentence'],row['item']+' '+row['word']))
    rating_text={r['id']:r['text'] for r in stream_rows(ROOT/'stimuli.jsonl')}
    previous={r['article']:rating_text[r['id']] for r in stream_rows(ROOT/'map.jsonl') if r['language']=='English'}
    tok=Tokenizer.from_file(str(TOKENIZER));docs=[];mapping=[];missing=set();all_words=set()
    for sid,rows in sorted(groups.items(),key=lambda pair:int(pair[0])):
        items={item for item,_,_ in rows};texts={s for _,s,_ in rows}
        if len(texts)!=1 or len(items)!=len(rows):raise ValueError('EEG sentence id or measured target repeats')
        text=texts.pop();words=list(re.finditer(r'\S+',text));positions={w.end():j for j,w in enumerate(words)}
        key=sha256(text.encode()).hexdigest();matches=[];is_changed=text!=previous[sid]
        for item,_,prefix in rows:
            if not text.startswith(prefix) or len(prefix) not in positions:raise ValueError('Original human EEG word has no matching boundary')
            j=positions[len(prefix)];same=previous[sid].startswith(prefix) and (len(prefix)==len(previous[sid]) or previous[sid][len(prefix)].isspace())
            matches.append([item,j,same]);all_words.add(item)
            if not same:missing.add(item)
        if not is_changed:continue
        enc=tok.encode(text,add_special_tokens=False);geom=[]
        for word in words:
            a,b=word.span();idx=[i for i,(l,h) in enumerate(enc.offsets) if l<b and h>a and text[max(a,l):min(b,h)].strip()]
            if not idx or enc.offsets[idx[-1]][1]!=b or tok.encode(text[:b],add_special_tokens=False).ids!=enc.ids[:idx[-1]+1]:raise ValueError('Original EEG word end is not committed')
            geom.append([a,b,idx[0],idx[-1]])
        docs.append({'id':key,'text':text,'words':geom});mapping.append({'id':key,'sentence':sid,'items':matches})
    if len(all_words)!=1726 or len(missing)!=180 or len(docs)!=27:raise ValueError('Original EEG overlap differs from registration')
    sealed_rows(ROOT/'erp-stimuli.jsonl',docs);sealed_rows(ROOT/'erp-map.jsonl',mapping)
    write_json(OUT/'erp-exposure-seal.json',dict(articles=27,new_original_exposures=180,all_eligible_noninitial_words=1726,identical_prefixes=1546,source=digest(ROOT/'outcomes.csv'),stimuli=digest(ROOT/'erp-stimuli.jsonl'),map=digest(ROOT/'erp-map.jsonl'),registration=digest(Path(__file__).with_name('convergent-word-geometry.md'))))


if __name__=='__main__':run()
