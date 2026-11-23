"""Recover the original 21-word Spanish exposure and the four printed response alternatives."""
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import csv,re,unicodedata,xml.etree.ElementTree as ET
from zipfile import ZipFile
from tokenizers import Tokenizer
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.identity_prepare import DATA,TOKENIZER,sealed_rows

ROOT=DATA/'semantic-clues';OUT=Path('research/results/reading-validation-spanish-clues');PROTOCOL=Path(__file__).with_name('spanish-clue-protocol.md');PUNCT='.,¿?¡!:;«»"\' '


def norm(s):return ''.join(c for c in unicodedata.normalize('NFD',s.casefold().strip(PUNCT)) if not unicodedata.combining(c))


def dictionary():
    z=ZipFile(ROOT/'wli_pseudowords.xlsx');ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    shared=[''.join(v.itertext()) for v in ET.fromstring(z.read('xl/sharedStrings.xml'))];rows=ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//m:row',ns);out={}
    for r in rows[1:]:
        v={}
        for c in r.findall('m:c',ns):
            el=c.find('m:v',ns);x=el.text if el is not None else '';v[re.sub(r'\d','',c.get('r'))]=shared[int(x)] if c.get('t')=='s' else x
        if v.get('B') and v.get('C'):out[norm(v['B'])]=v['C'].strip()
    return out


def logframes(p):
    b=p.read_bytes();s=b.decode('utf-16' if b[:2] in (b'\xff\xfe',b'\xfe\xff') else 'cp1252');subject=re.search(r'(?m)^\s*Subject:\s*(\d+)\s*$',s)
    records=[];block=0
    for f in re.findall(r'\*\*\* LogFrame Start \*\*\*(.*?)\*\*\* LogFrame End \*\*\*',s,re.S):
        data={k.strip():v.strip() for x in f.splitlines() if ':' in x for k,_,v in [x.strip().partition(':')]}
        if data.get('Running')!='MainTask':continue
        if data.get('Procedure')=='RestProc':block+=1
        elif data.get('Procedure') in ('TrialProc','TestProc'):records.append((block,data))
    return int(subject.group(1)) if subject else None,records


def build(p,lexicon):
    person,records=logframes(p);design=[];assign=[];errors=Counter();byblock=defaultdict(dict)
    for block,r in records:
        if r['Procedure']!='TrialProc':continue
        words=[r['Word'+s+str(i)].strip() for s in 'ABC' for i in range(1,8)];cs=r['TriggersFirstSentence'];cond={'11':'High','12':'Low','13':'Real'}.get(cs)
        if cond is None:errors['unknown_exposure_trigger_'+cs]+=1;continue
        if not all(words):errors['incomplete_triplet']+=1;continue
        text=' '.join(words);key=sha256(text.encode()).hexdigest();target=words[6].strip(PUNCT);category=words[17].strip(PUNCT);anchor=norm(target)
        if key not in byblock[block]:byblock[block][key]=(anchor,cond)
        design.append(dict(id=key,text=text,words=words,source=target,category=category,condition=cond,block=block))
    for block,r in records:
        if r['Procedure']!='TestProc':continue
        pseudo=r.get('TargetPseudoword','');options=[r['Word'+str(i)] for i in range(1,5)];target=r['CorrectWord'];slot=int(r['CorrectWordNumber'])-1
        if not 0<=slot<4 or norm(options[slot])!=norm(target) or any(not s for s in options):errors['invalid_choice_card']+=1;continue
        if norm(pseudo) not in lexicon or norm(lexicon[norm(pseudo)])!=norm(target):errors['unconfirmed_lexicon']+=1;continue
        eligible=[key for key,(x,c) in byblock[block].items() if x==norm(pseudo) and c in ('High','Low')]
        if len(eligible)!=1:errors['unmatched_exact_same_block']+=1;continue
        assign.append(dict(id=eligible[0],target=target,options=options,slot=slot,pseudo=pseudo,block=block))
    return person,design,assign,errors


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);tok=Tokenizer.from_file(str(TOKENIZER));lex=dictionary();docs={};cards=defaultdict(set);targets=defaultdict(set);exclusions=Counter();sources=read_json(ROOT/'log-source-sha.json');indices=set()
    with (ROOT/'wli_included_participants_list.csv').open(encoding='utf-8-sig') as f:
        inc=[r for r in csv.DictReader(f,delimiter=';')];included={int(r['id']) for r in inc if r['included'].casefold() in ('1','yes','true')}
    geometry={};sources_to_keep=[]
    for filename,expected in sorted(sources.items()):
        p=ROOT/'logs'/filename
        if digest(p)!=expected:raise ValueError('Source log checksum mismatch')
        person,design,choices,errors=build(p,lex)
        if person not in included or len(design)!=144:continue
        if person in indices:raise ValueError('Duplicate complete exposure session for reader')
        indices.add(person);sources_to_keep.append(filename);exclusions.update(errors)
        for r in design:
            if r['id'] in docs and docs[r['id']]!=r|{'block':docs[r['id']]['block']}:raise ValueError('Inconsistent original stimulus')
            docs[r['id']]=r
        for c in choices:
            cards[c['id']].add(tuple(c['options']));targets[c['id']].add(c['target'])
    if not indices:raise ValueError('Published inclusion column or actual source trials could not be reconciled')
    packet=[];mapping=[];vocab=set();spanerrors=Counter()
    def before_id(word):return tok.encode(' '+word.strip(PUNCT),add_special_tokens=False).ids[0]
    for k,r in sorted(docs.items()):
        target=r['source'];text=r['text'];probe='\nEn este texto, «'+target+'» significa';inputtext=text+probe;e=tok.encode(inputtext,add_special_tokens=False);raw=tok.encode(text,add_special_tokens=False);base=[];at=0
        if e.ids[:len(raw.ids)]!=raw.ids:raise ValueError('The readout suffix retokenizes original words')
        for word in r['words']:
            a=at;b=a+len(word);at=b+1;ii=[i for i,(l,u) in enumerate(raw.offsets) if (u>a and l<b) and text[max(a,l):min(b,u)].strip()]
            if not ii:raise ValueError('Original printed word cannot be indexed')
            base.append([a,b,ii[0],ii[-1]])
        cag=r['category'];w=r['words'][17];a,b,first,last=base[17];empty=0
        # Exclude punctuation affixed to the category wherever the tokenizer separates it.
        right=b-(len(w)-len(w.rstrip(PUNCT)));left=a+len(w)-len(w.lstrip(PUNCT));which=[i for i,(l,u) in enumerate(raw.offsets) if (u>left and l<right) and text[max(l,left):min(u,right)].strip()]
        if not which or raw.offsets[which[-1]][1]>right or raw.ids[which[0]]!=before_id(cag):spanerrors['uncertified_complete_category']+=1;continue
        accepted_cards=sorted(cards[k]);validtargets=sorted(targets[k]);
        if len(validtargets)>1:raise ValueError('One original text declares two conflicting target meanings')
        for card in accepted_cards:
            for x in card:vocab.add(before_id(x))
        vocab.add(raw.ids[which[0]]);fraction={'id':k,'text':inputtext,'article_end':len(raw.ids),'words':base,'category_tokens':[which[0],which[-1]+1]}
        packet.append(fraction);mapping.append({'id':k,'condition':r['condition'],'target_word':r['source'],'category':cag,'gold_onset':int(raw.ids[which[0]]),'cards':accepted_cards,'meaning':validtargets[0] if validtargets else lex.get(norm(r['source']),r['source']),'word_boundaries':base,'category_onsets':list(map(int,which)),'first':r['words'][:7],'middle':r['words'][7:14],'last':r['words'][14:],'article_tokens':len(raw.ids),'category_first':int(which[0])})
    sealed_rows(ROOT/'stimuli.jsonl',packet);sealed_rows(ROOT/'mapping.jsonl',mapping);vocab=sorted(vocab);write_json(ROOT/'vocabulary.json',{'onsets':vocab});write_json(ROOT/'eligible-logs.json',sources_to_keep)
    groups=defaultdict(dict)
    for d in mapping:
        if d['condition'] in ('High','Low'):groups[(norm(d['target_word']),tuple(d['first']),tuple(d['last']))][d['condition']]=d
    matched=[g for g in groups.values() if len(g)==2]
    write_json(OUT/'input-seal.json',{'protocol':digest(PROTOCOL),'material_sources':sources,'lexicon':digest(ROOT/'wli_pseudowords.xlsx'),'inclusion':digest(ROOT/'wli_included_participants_list.csv'),'eligible_logs':digest(ROOT/'eligible-logs.json'),'stimuli':digest(ROOT/'stimuli.jsonl'),'mapping':digest(ROOT/'mapping.jsonl'),'vocabulary':digest(ROOT/'vocabulary.json'),'published_included':len(included),'full_matched_participants':len(indices),'lexicon_entries':len(lex),'unique_triplets':len(packet),'conditions':dict(Counter(d['condition'] for d in mapping)),'same_first_last_groups':len(matched),'exact_native_length_pairs':sum(g['High']['article_tokens']==g['Low']['article_tokens'] for g in matched),'nondistinct_first_tokens_card_count':sum(len({before_id(x) for x in card})<4 for v in mapping for card in v['cards']),'missing_exposures':dict(exclusions),'category_coordinates':dict(spanerrors),'has_observed_any_response':False})
    print('participants',len(indices),'texts',len(packet),'same-first/last',len(matched),'repairs',dict(exclusions))

if __name__=='__main__':prepare()
