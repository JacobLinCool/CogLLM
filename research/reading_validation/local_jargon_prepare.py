"""Make an outcome-free stimulus and term-coordinate packet from Guo et al."""

import csv
from collections import defaultdict, Counter
import json
import re
from shutil import copyfile

from research.reading_validation.local_features import DATA, OUT
from research.reading_validation.local_forecast import WORD, digest


JARGON = DATA/'jargon'
REVISION = '22add4cc00274de199cc74bf0dbd629d05151b08'
SENTENCE = re.compile(r'(?<=[.!?])(?=\s+[A-Z0-9“"(])|(?<=\n)(?=\s*\S)')


def sentence_bounds(text):
    points = [0] + [m.start() for m in SENTENCE.finditer(text)] + [len(text)]
    return [(a,b) for a,b in zip(points,points[1:]) if WORD.search(text,a,b)]


def inside_term(text, term, segments):
    matches = list(re.finditer(r'(?<!\w)'+re.escape(term)+r'(?!\w)',text,re.IGNORECASE))
    if len(matches) != 1:
        return None,'absent' if not matches else 'ambiguous_occurrence'
    a,b = matches[0].span()
    for i,(lo,hi) in enumerate(segments):
        if lo <= a and b <= hi:
            words = list(WORD.finditer(text,lo,hi))
            starts = {w.start():n for n,w in enumerate(words)}
            ends = {w.end():n for n,w in enumerate(words)}
            if a not in starts or b not in ends:
                return None,'word_boundary'
            if ends[b]-starts[a]>=8:
                return None,'too_long'
            return (i,a,b),'included'
    return None,'crosses_sentence'


def prepare():
    raw = JARGON/'source.csv'
    papers, terms = {},defaultdict(set)
    with raw.open(newline='') as stream:
        for row in csv.DictReader(stream):
            publication, sentence, term = row['paper_sha'],row['abstract'],row['entity']
            if not all(isinstance(k,str) and k for k in (publication,sentence,term)):
                raise ValueError('Unexpected missing stimulus coordinates')
            if papers.setdefault(publication,sentence)!=sentence:
                raise ValueError('Paper identity maps to different texts')
            terms[publication].add(term)
    coverage=Counter()
    with (JARGON/'local-stimuli.jsonl').open('w') as stimuli, (JARGON/'geometry.jsonl').open('w') as geo:
        for paper, text in sorted(papers.items()):
            key='jargon/'+paper
            stimuli.write(json.dumps({'id':key,'text':text},ensure_ascii=False)+'\n')
            segments=sentence_bounds(text)
            spans=[]
            for t in sorted(terms[paper]):
                position,reason=inside_term(text,t,segments)
                coverage[reason]+=1
                if position is not None:
                    spans.append({'term':t,'sentence':position[0],'start':position[1],'end':position[2]})
            geo.write(json.dumps(dict(id=key,segments=segments,terms=spans),ensure_ascii=False)+'\n')
    copyfile(DATA/'local-lexical.npz',JARGON/'local-lexical.npz')
    result=dict(source_revision=REVISION,source_sha256=digest(raw),papers=len(papers),
                terms=sum(map(len,terms.values())),coverage=coverage,
                stimuli_sha256=digest(JARGON/'local-stimuli.jsonl'),
                lexical_sha256=digest(JARGON/'local-lexical.npz'),geometry_sha256=digest(JARGON/'geometry.jsonl'),
                protocol_sha256=digest(__import__('pathlib').Path(__file__).with_name('local-jargon-protocol.md')),
                used_fields=['entity','paper_sha','abstract'])
    target=OUT/'jargon'
    target.mkdir(exist_ok=True)
    (target/'source-seal.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    prepare()
