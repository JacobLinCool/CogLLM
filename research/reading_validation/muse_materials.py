"""Build a public German stimulus inventory and a complete answer-independent token alphabet."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import csv,io,json,unicodedata
from urllib.request import urlopen

from research.backbone_transfer.extract import digest,read_json,write_json,write_rows
from research.reading_validation.identity_colab import DATA
from research.token_readout.data import TOKENIZER,TOKENIZER_SHA256
from tokenizers import Tokenizer

WORK=DATA/'muse';OUT=Path('research/results/reading-validation-muse')
PLAN=Path(__file__).with_name('muse-protocol.md')
RAW='fabaf98e7932e68a9b2c05e2378565141d4d9e8645c070645258ffd8bc5431a2'


def canonical(word):
    return unicodedata.normalize('NFC',word).strip().lower().translate(str.maketrans({'ä':'ae','ö':'oe','ü':'ue','ß':'ss'}))


def original():
    data=urlopen('https://osf.io/download/nb3u4/',timeout=60).read()
    if sha256(data).hexdigest()!=RAW:raise ValueError('Original German response table changed')
    return csv.DictReader(io.StringIO(data.decode('utf-8-sig')))


def lexical():
    from wordfreq import top_n_list,word_frequency
    if version('wordfreq')!='3.1.1' or digest(TOKENIZER)!=TOKENIZER_SHA256:raise ValueError('Changed independent lexicon')
    t=Tokenizer.from_file(str(TOKENIZER));words=[];selected={};aliases=defaultdict(set);surfaces=defaultdict(set)
    freq=Counter();collisions=set()
    for original_word in top_n_list('de',100000):
        w=unicodedata.normalize('NFC', original_word).lower(); key=canonical(w)
        if not w.isalpha() or not key:continue
        if key not in selected and len(selected)>=4096:continue
        candidates=set()
        for printed in (w,w[0].upper()+w[1:]):
            target=' '+printed;z=t.encode(target,add_special_tokens=False).ids
            if len(z)==1 and t.decode(z,skip_special_tokens=False)==target:candidates.add(z[0])
        if not candidates:continue
        if key not in selected:selected[key]=len(words);words.append(key)
        if w in surfaces[key]:continue
        surfaces[key].add(w);aliases[key]|=candidates
        freq[key]+=word_frequency(w,'de')
        if len(surfaces[key])>1:collisions.add(key)
    if len(words)!=4096:raise ValueError(f'Insufficient certified German whole-word inventory: {len(words)}')
    if sum(map(len,aliases.values()))!=len(set.union(*aliases.values())):raise ValueError('Token assigned to two independent spellings')
    out=[dict(key=w,tokens=sorted(aliases[w]),freq=freq[w],surface=sorted(surfaces[w])) for w in words]
    path=WORK/'dictionary.json';write_json(path,out)
    print('Independent German lexicon',len(words),sum(len(a) for a in aliases.values()),len(collisions),flush=True)


def geometry():
    people={};readers={};seen=set();present=defaultdict(set);vis=Counter()
    for row in original():
        source=row['participant_private_id'];mode=source[0]
        if mode not in ('a','v'):raise ValueError('Original cohort key outside registered data')
        person=people.setdefault(source,len(people));readers[person]=mode
        text=row['sentence'];seen.add(text);present[mode].add(text);vis[mode]+=1
    if Counter(readers.values())!={'v':98,'a':133} or len(seen)!=619:raise ValueError('Cohort identity mismatch')
    material={s:sha256(s.encode()).hexdigest() for s in sorted(seen)}
    packet=WORK/'stimuli.jsonl';write_rows(packet,sorted([dict(id=k,text=s) for s,k in material.items()],key=lambda r:r['id']))
    study={mode:sorted(material[s] for s in v) for mode,v in present.items()}
    write_json(WORK/'cohorts.json',{'readers':readers,'stems':study})
    t=Tokenizer.from_file(str(TOKENIZER))
    for s in seen:
        offs=t.encode(s,add_special_tokens=False).offsets
        if not offs or offs[-1][1]!=len(s):raise ValueError('Final visible German word not certified')
    d=read_json(WORK/'dictionary.json')
    write_json(OUT/'exposure-seal.json',dict(protocol=digest(PLAN),raw=RAW,materials=digest(packet),
        lexicon=digest(WORK/'dictionary.json'),native_tokenizer=TOKENIZER_SHA256,cohorts=digest(WORK/'cohorts.json'),
        written_people=98,auditory_people=133,written_exposures=vis['v'],auditory_exposures=vis['a'],
        stimuli=619,written=len(present['v']),auditory=len(present['a']),shared=len(present['v']&present['a']),
        written_only=len(present['v']-present['a']),named_categories=len(d),absorbed_native_tokens=sum(len(e['tokens']) for e in d),
        spelling_transliteration_collisions=sum(len(e['surface'])>1 for e in d),person_release_identifiers_saved=False,
        answer_fields_read=False,original_sources=['https://osf.io/download/nb3u4/','https://osf.io/download/ymvxj/']))
    print(read_json(OUT/'exposure-seal.json'),flush=True)


def select(phase, source=None):
    design=read_json(OUT/'exposure-seal.json')
    if design['protocol']!=digest(PLAN):raise ValueError('Changed source protocol')
    lookup={e['key']:j for j,e in enumerate(read_json(WORK/'dictionary.json'))}
    pool=read_json(WORK/'cohorts.json')
    source_items=set(pool['stems']['a']);write_items=set(pool['stems']['v']);shared=source_items&write_items
    people={};rows=[];eligible=Counter();seen=set()
    if phase=='target' and not (OUT/'human-forecast-lock.json').exists():raise ValueError('Lock every external personal forecast first')
    for row in (original() if source is None else source):
        code=row['participant_private_id'];r=people.setdefault(code,len(people));mode=code[0]
        x=sha256(row['sentence'].encode()).hexdigest()
        if phase=='training' and mode!='a':continue
        if phase=='history' and (mode!='v' or x not in shared):continue
        if phase=='target' and (mode!='v' or x in source_items):continue
        key=r,x
        if key in seen:raise ValueError('A participant appears twice at a held sentence')
        seen.add(key)
        v=canonical(row['response']);gr=row['grammar']
        if not v or v in ('999','na') or gr=='777':
            eligible['omissions_or_grammar']+=1;continue
        rows.append([r,x,lookup.get(v,len(lookup))])
    file=WORK/(phase+'-observations.json');write_json(file,rows)
    print(phase,len(rows),len({r for r,_,_ in rows}),len({x for _,x,_ in rows}),dict(eligible),flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('action',choices=['lexical','geometry','source','training','history','target'])
    a=p.parse_args().action
    if a=='lexical':lexical()
    elif a=='geometry':geometry()
    elif a=='source':
        material=list(original())
        select('training',iter(material));select('history',iter(material))
    else:select(a)
