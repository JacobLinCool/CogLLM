"""Certify identical visible original word endpoints for the three pretrained encoders."""
from pathlib import Path
from hashlib import sha256
import re
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.identity_prepare import sealed_rows,DATA
from research.reading_validation.transition_architecture_prepare import load_tokenizer,word_indices,CANDIDATES

OUT=Path('research/results/reading-validation-physiology-portability')
PLAN=Path(__file__).with_name('physiology-portability-protocol.md')
ROOT=DATA/'physiology-portability'


def build():
    ROOT.mkdir(exist_ok=True);sources=[DATA/'convergent/stimuli.jsonl',DATA/'convergent/erp-stimuli.jsonl',DATA/'multilab/stimuli.jsonl'];spec=read_json(Path('research/backbone_transfer/models.json'))['models'];tokens={c:load_tokenizer(c,spec[c]) for c in CANDIDATES}
    old={};docs=[];rejections=[]
    for p in sources:
        for row in stream_rows(p):
            k=row['id'];text=row['text'];words=[list(m.span()) for m in re.finditer(r'\S+',text)]
            if sha256(text.encode()).hexdigest()!=k or words!=[w[:2] for w in row['words']]:raise ValueError('Original source print coordinates changed')
            if k in old:
                if row!=old[k]:raise ValueError('Conflicting original variants')
                continue
            old[k]=row;geometry={};sizes={}
            for name,tok in tokens.items():
                enc=tok.encode(text,add_special_tokens=False);ix=word_indices(text,words,tok,enc);sizes[name]=len(enc.ids);geometry[name]=ix
                if any(i is None for i in ix):rejections.append([k,name,[i for i,idx in enumerate(ix) if idx is None]])
            if any(v is None for group in geometry.values() for v in group):continue
            if geometry['qwen']!=[w[2:] for w in row['words']]:raise ValueError('Original Qwen prefix positions differ')
            docs.append(dict(id=k,text=text,words=words,native=geometry,tokens=sizes))
    sealed_rows(ROOT/'stimuli.jsonl',docs);write_json(OUT/'input-seal.json',dict(protocol=digest(PLAN),source=[digest(p) for p in sources],stimuli=digest(ROOT/'stimuli.jsonl'),unique_texts=len(old),complete_texts=len(docs),word_ends=sum(len(r['words']) for r in docs),native_tokens={k:sum(r['tokens'][k] for r in docs) for k in CANDIDATES},excluded=rejections));print('Three-encoder common geometry',len(docs),len(rejections))


if __name__=='__main__':build()
