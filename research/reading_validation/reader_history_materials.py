"""Isolate each published sentence stem without consuming any answer columns."""
import csv,json,re
from hashlib import sha256
from pathlib import Path
from tokenizers import Tokenizer
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json,write_rows
from research.reading_validation.identity_prepare import DATA,TOKENIZER
from research.reading_validation.cloze_population_data import ROOT as POP
DIR=DATA/'reader-history'
OUT=Path('research/results/reading-validation-reader-history')
PLAN=Path(__file__).with_name('reader-history-protocol.md')


def normalized(text):return " ".join(re.findall(r"[\w']+",text.casefold()))


def main():
    seal=read_json(OUT/'registration.json')
    if seal['protocol']!=digest(PLAN):raise ValueError("Different reader-history design")
    texts=read_json(POP/'geometry.json')
    material={r['id']:r for r in stream_rows(POP/'semantic-stimuli.jsonl')}
    source_prefixes={normalized(material[did]['text'][:material[did]['ends'][i['word']-1][1]]) for did,g in texts.items()
                     if g['source']=='UCL' for i in g['items']}
    tok=Tokenizer.from_file(str(TOKENIZER))
    packet=[];mapping={};exclude=[]
    with (DIR/'output.tsv').open() as f:
        for row in csv.DictReader(f,delimiter='\t'):
            original=row['Question']
            src=re.fullmatch(r"(.+?)\s+_+\.",original)
            if not src:raise ValueError("Unverifiable original display")
            text=src[1]
            if normalized(text) in source_prefixes:exclude.append(original);continue
            uid=sha256(text.encode()).hexdigest()
            enc=tok.encode(text,add_special_tokens=False)
            if enc.offsets[-1][1]!=len(text) or not len(enc.ids):raise ValueError("Missing final original word")
            if original in mapping:raise ValueError("Duplicate original stem")
            mapping[original]=uid;packet.append({'id':uid,'text':text})
    if len(set(mapping.values()))!=len(mapping) or len(mapping)+len(exclude)!=3085:raise ValueError("Original stems differ")
    write_rows(DIR/'trial-stimuli.jsonl',sorted(packet,key=lambda r:r['id']))
    write_json(DIR/'question-map.json',mapping)
    write_json(OUT/'input-seal.json',{'protocol':digest(PLAN),'packet':digest(DIR/'trial-stimuli.jsonl'),
               'map':digest(DIR/'question-map.json'),'source_file':digest(DIR/'output.tsv'),
                'all_published_stems':3085,'eligible_stems':len(packet),'overlapping_UCL':len(exclude),
                'examined_fields':['Question'],'exposure':'Original sentence preceding the blank. People saw the blank followed by a period.'})
    print(len(packet),len(exclude))


if __name__=='__main__':main()
