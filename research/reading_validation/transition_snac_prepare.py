"""Build an outcome-blind, source-aligned SNaC packet with intact summary grouping."""
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

from research.reading_validation.transition_external_prepare import ROOT

RESULTS=Path('research/results/reading-validation-transition-snac')
SOURCE=ROOT/'snac-source.json'
DIGEST='c4d238bfd365a107fa4f986d33feb44f0f37c7fc85207a21096313a44f786ef4'
WORD=re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|[0-9]+(?:[.,][0-9]+)*")


def digest(s): return sha256(s.encode()).hexdigest()


def source():
    raw=SOURCE.read_bytes()
    if sha256(raw).hexdigest()!=DIGEST:raise ValueError('Unexpected source archive')
    return json.loads(raw)


def download():
    SOURCE.parent.mkdir(parents=True,exist_ok=True)
    temp=SOURCE.with_suffix('.part')
    subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','3',
                    'https://drive.usercontent.google.com/download?id=1ff-pV2sX9XNDMdaPxY7v22T2i0235tcE&export=download&confirm=t',
                    '-o',str(temp)],check=True)
    if sha256(temp.read_bytes()).hexdigest()!=DIGEST:
        raise ValueError('The downloaded file differs from the original source release')
    temp.replace(SOURCE)


def prepare():
    from tokenizers import Tokenizer
    snapshot=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer.json'
    tok=Tokenizer.from_file(str(snapshot)); docs=[]; units=[];omitted=[];hashes={};trim=0
    for doc,segments in source().items():
        m=re.fullmatch(r'(book_175b|book_6b|tripod)(\d+)',doc)
        if not m or set(map(int,segments))!=set(range(len(segments))):raise ValueError('Unknown source grouping')
        group=('movie-' if m[1]=='tripod' else 'book-')+m[2];blocks=[];records=[];pos=0
        for i in range(len(segments)):
            text=' '.join(segments[str(i)]['text'].split())
            if not text:raise ValueError('Empty source segment')
            records.append({'index':i,'start':pos,'end':pos+len(text)});blocks.append(text);pos+=len(text)+1
        text=' '.join(blocks);enc=tok.encode(text,add_special_tokens=False);n=len(enc.ids)
        if n>2048:
            omitted.append({'doc':doc,'group':group,'tokens':n});continue
        token_ends={e:j for j,(_,e) in enumerate(enc.offsets)}
        words=[]
        for segment in records:
            for j,w in enumerate(WORD.finditer(text,segment['start'],segment['end'])):
                end=token_ends.get(w.end());start=next((k for k,(a,b) in enumerate(enc.offsets) if b>w.start() and a<w.end()),None)
                if end is None or start is None or w.start()<enc.offsets[start][0] or enc.offsets[end][1]!=w.end():
                    trim+=1;continue
                if tok.encode(text[:w.end()],add_special_tokens=False).ids!=enc.ids[:end+1]:
                    trim+=1;continue
                words.append([w.start(),w.end(),start,end,segment['index'],j])
        docs.append({'id':doc,'text':text,'words':words});units.append({'doc':doc,'group':group,'collection':m[1],
            'segments':records,'words':len(words),'tokens':n,'text_sha256':digest(text)})
        role='test' if m[1]=='tripod' else 'books'
        hashes.setdefault(role,set()).update(digest(' '.join(text[:seg['end']].casefold().split())) for seg in records)
    if hashes['books'] & hashes['test']:raise ValueError('A complete source prefix crossed the train/test split')
    families=sorted({r['group'] for r in units if r['collection']!='tripod'},key=lambda g:digest('cogllm-snac-books-v1:'+g))
    train=set(families[:int(.8*len(families))]);
    for r in units:r['split']='test' if r['collection']=='tripod' else 'train' if r['group'] in train else 'dev'
    path=ROOT/'snac-stimuli.jsonl';path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in docs))
    RESULTS.mkdir(exist_ok=True,parents=True)
    manifest={'source_sha256':DIGEST,'packet_sha256':sha256(path.read_bytes()).hexdigest(),
        'protocol_sha256':sha256(Path('research/reading_validation/transition-snac-protocol.md').read_bytes()).hexdigest(),
        'excluded_complete_summaries':omitted,'uncertified_words':trim,'books_to_movies_exact_prefixes':0,'documents':units}
    (RESULTS/'stimuli.json').write_text(json.dumps(manifest,indent=2)+'\n');print(len(docs),sum(r['words'] for r in units),trim,omitted,manifest['packet_sha256'])

if __name__=='__main__':
    if not SOURCE.exists():
        download()
    prepare()
