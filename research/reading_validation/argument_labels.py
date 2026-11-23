"""Align source judgments with their verified character ranges, retaining unknown cases."""
from collections import Counter,defaultdict
import csv
from io import BytesIO,StringIO
import json
import re
from zipfile import ZipFile

from research.reading_validation.argument_data import OUT,WORK,author_rows,sentences
from research.reading_validation.memory_data import load_rows,sha

LEVELS={'Ineffective':0,'Adequate':1,'Effective':2}


def normalized(text):return ' '.join(text.split())


def align(text,span,original):
    literal=original.strip()
    if not literal:return None
    a,b=span
    if 0<=a<b<=len(text) and normalized(text[a:b])==normalized(literal):
        return a+len(text[a:b])-len(text[a:b].lstrip()),b-len(text[a:b])+len(text[a:b].rstrip())
    pattern=r'\s+'.join(map(re.escape,literal.split()))
    matches=list(re.finditer(pattern,text))
    if len(matches)==1:return matches[0].span()
    return None


def region(sentences_,span):
    if span is None:return []
    a,b=span
    return [i for i,(s,e) in enumerate(sentences_) if a<e and b>s]


def annotations(components,edges,coords):
    links=defaultdict(set);kinds=defaultdict(lambda:defaultdict(set));grades=defaultdict(list)
    count=Counter();excluded=set();nonlocalizable=False
    for i,item in components.items():
        if item['grade'] is None:continue
        here=region(coords,item['span'])
        if here:grades[here[-1]].append(item['grade'])
        else:count['missing_effectiveness_position']+=1
    for s,parent,kind in edges:
        count[kind+'_total']+=1
        curr=region(coords,components[s]['span'])
        if not curr:
            nonlocalizable=True;count['unlocalized_child']+=1;continue
        pos=curr[-1];target=region(coords,components[parent]['span']) if parent is not None else []
        if not target:excluded.add(pos);count['unlocalized_or_ambiguous_parent']+=1;continue
        if components[parent]['span'][1]>components[s]['span'][0]:
            count['intra_or_forward']+=1;continue
        allowed={i for i in target if i<pos}
        if not allowed:count['same_sentence']+=1;continue
        links[pos].update(allowed);kinds[kind][pos].update(allowed);count[kind+'_backward']+=1
    return dict(links={k:sorted(v) for k,v in links.items()},types={t:{k:sorted(v) for k,v in l.items()} for t,l in kinds.items()},
               grades=dict(grades),exclude=sorted(excluded-set(links)),complete=not nonlocalizable,audit=dict(count))


def source_labels(records,article):
    components={};ordered=[];lookup=defaultdict(list);edges=[]
    for r in records:
        if r['discourse_type']=='Unannotated':continue
        i=len(ordered);ordered.append(r)
        span=align(article,(int(float(r['discourse_start'])),int(float(r['discourse_end']))),r['discourse_text'])
        grade=r['discourse_effectiveness']
        if grade and grade not in LEVELS:raise ValueError(f'Unknown effectiveness level {grade}')
        components[i]=dict(span=span,grade=LEVELS.get(grade))
        lookup[r['discourse_type'],normalized(r['discourse_text'])].append(i)
    for i,r in enumerate(ordered):
        if not r['hierarchical_text'].strip():continue
        target=lookup[r['hierarchical_label'],normalized(r['hierarchical_text'])]
        parent=target[0] if len(target)==1 else None
        edges.append((i,parent,'hierarchy'))
    return annotations(components,edges,sentences(article))


def brat(text,source):
    components={};edges=[]
    for row in source.splitlines():
        fields=row.split('\t')
        if fields[0].startswith('T'):
            kind,a,b=fields[1].split();a,b=int(a),int(b)
            if text[a:b]!=fields[2]:raise ValueError('BRAT span disagrees with the original essay characters')
            components[fields[0]]=dict(span=(a,b),grade=None,kind=kind)
        elif fields[0].startswith('R'):
            kind,s,t=fields[1].split();s=s.removeprefix('Arg1:');t=t.removeprefix('Arg2:')
            if kind not in ('supports','attacks'):raise ValueError('Unknown explicit relation')
            edges.append((s,t,kind))
    return annotations(components,edges,sentences(text))


def bank(role):
    if role not in ('train','dev','held','external'):raise ValueError('Unknown source partition')
    meta=json.loads((OUT/'inputs.json').read_bytes())
    if any(sha(WORK/k)!=v for k,v in meta['raw'].items()):raise ValueError('The original release has changed')
    texts={r['id']:r['text'] for r in load_rows(WORK/'stimuli.jsonl')}
    material={r['id']:k for k,r in meta['articles'].items() if r['role']==role}
    if role=='external':
        with ZipFile(WORK/'relations.zip') as outer,ZipFile(BytesIO(outer.read('ArgumentAnnotatedEssays-2.0/brat-project-final.zip'))) as z:
            return {k:brat(texts[k],z.read(f'brat-project-final/essay{i:03d}.ann').decode()) for i,k in material.items()}
    selected=defaultdict(list)
    for r in author_rows():
        if (k:=material.get(r['essay_id_comp'])) is not None:
            selected[k].append({name:r[name] for name in ('discourse_type','discourse_start','discourse_end','discourse_text',
                                               'discourse_effectiveness','hierarchical_label','hierarchical_text')})
    return {k:source_labels(records,texts[k]) for k,records in selected.items()}


def fingerprint(text):
    # Require a unique original paragraph and all ASCII letter/digit sequences in the same order.
    # Tokenization can insert spaces before "'s" or around hyphens. No statistical fuzzy matching.
    return ''.join(re.findall('[a-z0-9]+',text.casefold()))


def sufficiency():
    meta=json.loads((OUT/'inputs.json').read_bytes())
    materials={r['id']:k for k,r in meta['articles'].items() if r['role']=='external'}
    texts={r['id']:r['text'] for r in load_rows(WORK/'stimuli.jsonl')}
    with ZipFile(WORK/'votes.zip') as z:
        with StringIO(z.read('Annotations/annotations.csv').decode('utf8'),newline='') as f:
            votes={(int(r['ESSAY']),int(r['ARGUMENT'])):(fingerprint(r['TEXT']),[int(r[c]=='Sufficiency') for c in ('A1','A2','A3')])
                for r in csv.DictReader(f,delimiter=';')}
    with ZipFile(WORK/'sufficiency.zip') as z:
        # An authors' tokenization issue affects non-ASCII punctuation, which fingerprint omits.
        with StringIO(z.read('UKP-InsufficientArguments_v1.0/data-tokenized.tsv').decode('latin1'),newline='') as f:
            rows=list(csv.DictReader(f,delimiter='\t'))
    if len(rows)!=1029 or sum(r['ANNOTATION']=='insufficient' for r in rows)!=348 or len(votes)!=433:
        raise ValueError('The author-published number of majority and individual outcomes changed')
    out=[];audit=Counter()
    for r in rows:
        pair=int(r['ESSAY']),int(r['ARGUMENT']);key=materials.get(pair[0])
        if key is None:audit['ineligible_article']+=1;continue
        t=texts[key];gold=fingerprint(r['TEXT'])
        matched=[m.span() for m in re.finditer(r'[^\r\n]+',t) if gold and fingerprint(m.group())==gold]
        if len(matched)!=1:audit['no_unique_original_paragraph']+=1;continue
        label=int(r['ANNOTATION']=='insufficient');signed=votes.get(pair)
        vs=signed[1] if signed else None
        if signed and signed[0]!=gold:raise ValueError('Raters received a different paragraph')
        if vs is not None and int(sum(vs)>=2)!=label:audit['published_label_differs_from_vote_majority']+=1
        regions=region(sentences(t),matched[0])
        out.append(dict(article=key,positions=regions,label=label,votes=vs))
    return out,dict(audit)
