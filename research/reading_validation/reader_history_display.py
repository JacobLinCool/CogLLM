"""Certify each stimulus spelling against the original fields actually presented."""
import csv,re
from pathlib import Path
from hashlib import sha256
from collections import Counter
from tokenizers import Tokenizer
from research.backbone_transfer.extract import digest,read_json,write_json,write_rows
from research.reading_validation.reader_history_materials import DIR,OUT,TOKENIZER

def run():
    published=read_json(DIR/'question-map.json')
    actual=set()
    for name in ('anonymized_fill-in-1x35.csv','anonymized_fill-in-1x50.csv','anonymized_fill-in-61x50.csv'):
        with (DIR/name).open() as fp:
            for r in csv.DictReader(fp):actual.update(r[k] for k in r if k.startswith('Input.sentence'))
    u=sorted(actual-set(published));v=sorted(set(published)-actual)
    if len(actual)!=3085 or len(u)!=1 or len(v)!=1 or u[0].strip('"').replace('" ','"" ',1)!=v[0]:
        raise ValueError("Unexpected difference between summarized and original displayed wording")
    mapping={};rows=[];t=Tokenizer.from_file(str(TOKENIZER))
    for q in sorted(actual):
        src=re.fullmatch(r"(.+?)\s+_+\.",q)
        if not src:raise ValueError("Unresolved original question form")
        text=src[1];uid=sha256(text.encode()).hexdigest()
        if t.encode(text,add_special_tokens=False).offsets[-1][1]!=len(text):raise ValueError("Missing original word")
        mapping[q]=uid;rows.append({'id':uid,'text':text})
    file=DIR/'verified-trial-stimuli.jsonl';write_rows(file,sorted(rows,key=lambda r:r['id']))
    geo=DIR/'verified-question-map.json';write_json(geo,mapping)
    write_json(OUT/'verified-input-seal.json',{'protocol':read_json(OUT/'registration.json')['protocol'],
         'previous':digest(OUT/'input-seal.json'),'packet':digest(file),'map':digest(geo),'official_stems':len(mapping),
         'forms_changed':len(u),'observed_raw_field':u,'summary_table_field':v,'examined_raw_columns':['Input.sentenceNN']})
    print('Verified original',len(actual),'recovered',len(u))
if __name__=='__main__':run()
