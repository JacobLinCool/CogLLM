"""Prepare anonymized, source-numbered quiz words and recorded correct-buzz risk sets."""

import bisect
import csv
from collections import Counter, defaultdict
from hashlib import sha256
from html.parser import HTMLParser
import re
import tarfile
from pathlib import Path

from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.reading_validation.identity_prepare import DATA, TOKENIZER, sealed_rows

ROOT = DATA/'recognition'
OUT = Path('research/results/reading-validation-recognition')
PROTOCOL = Path(__file__).with_name('recognition-protocol.md')
SOURCE = Path('.cache/cogllm/semantic-tracking/quizbowl-open-data-audit')
EXTERNAL = Path('.cache/cogllm/semantic-tracking/qanta25-source-audit')


def key(s):
    return sha256(s.encode()).hexdigest()


class Marked(HTMLParser):
    def __init__(self):
        super().__init__(); self.bits=[];self.cursor=0;self.entries=[]

    def handle_starttag(self,tag,attrs):
        if tag=='m': self.entries.append((int(dict(attrs)['v']),self.cursor))

    def handle_data(self,data):
        self.bits.append(data);self.cursor+=len(data)


def tagged(body):
    source=Marked();source.feed(body);source.close();raw=''.join(source.bits);marks=source.entries
    if len(marks)<2 or [i for i,_ in marks]!=list(range(len(marks))):return None
    stem=raw[:marks[1][1]].strip();match=re.fullmatch(r'(\d+)\.',stem)
    if not match:return None
    where=marks[1][1];text=raw[where:].rstrip();starts=[i-where for _,i in marks[1:]];ends=[]
    for a,b in zip(starts,starts[1:]+[len(text)],strict=True):
        b=a+len(text[a:b].rstrip())
        if b<=a:return None
        ends.append(b)
    return int(match.group(1)),text,ends


def tokens(text,ends,tokenizer):
    encoded=tokenizer.encode(text,add_special_tokens=False);starts=[a for a,b in encoded.offsets];res=[];old=0
    for stop in ends:
        i=bisect.bisect_left(starts,stop)-1
        if i<0 or encoded.offsets[i][1]>stop or res and i<=res[-1] or not old<stop:return None
        res.append(i);old=stop
    return res if res and len(encoded.ids)<=1024 else None


def external(tokenizer,catalog):
    questions={r['qid']:r for r in stream_rows(EXTERNAL/'tossup_questions.jsonl')};ids={};bins={};ann=defaultdict(list);excl=Counter()
    for k,r in questions.items():
        text=r['question'];ending=[];spans=[]
        for a,b in r['clue_spans']:
            words=list(re.finditer(r'\S+',text[a:b]));q=len(ending);ending.extend(a+w.end() for w in words);spans.append((q,len(ending)))
        t=tokens(text,ending,tokenizer)
        if t is None:excl['native_unsupported_question']+=1;continue
        uid=key(text);ids[k]=uid;bins[k]=spans
        catalog[uid]={'id':uid,'text':text,'word_char_ends':ending,'word_token_ends':t}
    seen=set()
    for f in EXTERNAL.glob('tossup_responses-*.json'):
        for v in read_json(f):
            ident=v['response_id'];k=v['qid']
            if ident in seen:raise ValueError('Repeated QANTA event')
            seen.add(ident)
            if not v['player_id'].startswith('h-') and not v['player_id'].startswith('h_'):continue
            if int(v['correct'])!=1:continue
            stop=v['buzz_pos']; num=v['buzz_clue_index'];boundaries=questions[k]['clue_token_indices']
            if not (1<=stop<=boundaries[-1] and 1+bisect.bisect_left(boundaries,stop)==num):excl['position']+=1;continue
            if k not in ids:excl['native_unsupported_event']+=1;continue
            ann[k].append(bins[k][num-1])
    grades=[]
    for k,events in sorted(ann.items()):
        labels=Counter(events);grades.append({'id':ids[k],'group':key(k),'source':'QANTA25','intervals':[[a,b,n] for (a,b),n in sorted(labels.items())],
              'n':len(events),'subject':str(questions[k]['metadata'].get('category','')),'packet':questions[k]['metadata'].get('packet','')})
    if len(seen)!=827 or sum(len(v) for v in ann.values())+excl['native_unsupported_event']!=155:raise ValueError(f'QANTA25 event join disagrees with earlier audit: {len(seen)}, {sum(map(len,ann.values()))}, {excl}')
    return grades,dict(excl)


def prepare():
    ROOT.mkdir(exist_ok=True,parents=True);OUT.mkdir(exist_ok=True,parents=True);tok=Tokenizer.from_file(str(TOKENIZER));docs={};by_edition={};excluded=Counter()
    src=SOURCE/'open-data-5d86399.tar.gz'
    with tarfile.open(src) as z:
        for f in z:
            if not f.name.endswith('.w.html'):continue
            day,name=f.name.split('/')[-2:];packet=name.split('.')[0];s=z.extractfile(f).read().decode()
            for body in re.findall(r'<p\b[^>]*class="[^"]*\btu\b[^"]*"[^>]*>(.*?)</p>',s,re.S):
                v=tagged(body)
                if not v:excluded['unsupported_marked_paragraph']+=1;continue
                no,text,ends=v;ck=(day,packet,str(no));t=tokens(text,ends,tok)
                if t is None:excluded['no_native_word_boundary']+=1;continue
                if ck in by_edition:raise ValueError('Conflicting numbered tossup')
                ident=key(text);row={'id':ident,'text':text,'word_char_ends':ends,'word_token_ends':t}
                if ident in docs and row!=docs[ident]:raise ValueError('Conflicting normalized word markings')
                by_edition[ck]=ident;docs[ident]=row
    seen=set();answers=defaultdict(set);lineages=defaultdict(set);category=defaultdict(set);counts=defaultdict(Counter)
    with (SOURCE/'tossups.tsv').open(newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            day,pack,num=r['question_set_edition'],r['packet'],r['tossup'];k=by_edition.get((day,pack,num))
            if k is None:excluded['event_no_certified_source']+=1;continue
            if r['words']!=str(len(docs[k]['word_char_ends'])):excluded['word_count_mismatch']+=1;continue
            if not r['answer'].strip():excluded['no_answer_identity']+=1;continue
            answers[k].add(' '.join(r['answer'].casefold().split()));lineages[k].add(pack+'-'+num);category[k].add(r['category'])
            identity=(day,r['tournament'],r['room'],r['round'],num,r['team'],r['player'])
            if identity in seen:excluded['duplicate_logged_player_turn']+=1;continue
            seen.add(identity)
            if r['protested']:excluded['protest']+=1;continue
            if r['bounceback']:excluded['bounceback']+=1;continue
            if r['buzz_value']!='10':excluded['uncounted_incorrect']+=1;continue
            loc=r['buzz_location']
            if not loc.isdigit() or not 1<=int(loc)<=len(docs[k]['word_char_ends']):excluded['unknown_buzz_word']+=1;continue
            counts[k][int(loc)-1]+=1
    parents={k:k for k in counts}
    def ancestor(k):
        while k!=parents[k]:parents[k]=parents[parents[k]];k=parents[k]
        return k
    seenkeys={}
    for k in sorted(counts):
        for label in [*('a/'+v for v in answers[k]),*('l/'+v for v in lineages[k])]:
            if label in seenkeys:
                s,t=ancestor(k),ancestor(seenkeys[label]);parents[max(s,t)]=min(s,t)
            else:seenkeys[label]=k
    grades=defaultdict(list);catalog={k:docs[k] for k in counts};parts={}
    for k,v in sorted(counts.items()):
        family=ancestor(k);num=int(key('20261122-recognition/'+family)[:10],16)%100;part='train' if num<60 else 'validation' if num<80 else 'test'
        parts[family]=part;word_rows=[[i,i+1,qty] for i,qty in sorted(v.items())]
        grades[part].append({'id':k,'group':family,'source':'ACF','intervals':word_rows,'n':sum(v.values()),'subject':','.join(sorted(category[k])),
                             'packet':','.join(sorted({v.split('-')[0] for v in lineages[k]}))})
    if len(catalog)<150 or set(grades)!={'train','validation','test'}:raise ValueError('Insufficient certified ACF sources')
    external_grades,external_exclusions=external(tok,catalog);grades['external']=external_grades
    sealed_rows(ROOT/'stimuli.jsonl',[catalog[k] for k in sorted(catalog)]); hashes={}
    for part,records in grades.items():
        path=ROOT/(part+'.jsonl');sealed_rows(path,records);hashes[part]=digest(path)
    seal={'registered_protocol':digest(PROTOCOL),'acf_archive':digest(src),'acf_responses':digest(SOURCE/'tossups.tsv'),'stimuli_sha256':digest(ROOT/'stimuli.jsonl'),
        'outcome_sha256':hashes,'exclusions':dict(excluded),'external_exclusions':external_exclusions,
        'counts':{k:{'texts':len(v),'families':len({r['group'] for r in v}),'observed_correct_events':sum(r['n'] for r in v)} for k,v in grades.items()},
        'source_assay':'word-recorded, originally oral competitive recognition conditional on a correct recorded first opportunity','texts_to_accelerator':len(catalog)}
    write_json(OUT/'input-seal.json',seal);print(seal['counts'],seal['exclusions'],flush=True)


if __name__=='__main__':prepare()
