"""Compute causal linguistic competitors to a word-level antecedent pointer."""
from collections import Counter
import math
import re

import numpy as np


FORMS={'she':'female','her':'female','hers':'female','herself':'female','he':'male','him':'male','his':'male','himself':'male',
       'they':'plural','them':'plural','their':'plural','theirs':'plural','themselves':'plural','themself':'plural',
       'it':'neuter','its':'neuter','itself':'neuter'}
PARTS=('NNP','NNPS','NN','NNS','PRP','PRP$','DT','VB','VBD','VBN','IN','JJ','CD','CC','RB','other')
POSITION=('log_words_back','inverse_word_distance','within_5','within_20','within_80','log_sentences_back','same_sentence')
LEXICAL=('log_word_length','uppercase_initial','uppercase_all','numeric','log_previous_occurrences','last_same_form','same_query_form',
         'candidate_pronoun','pronoun_class_matches','pronoun_class_conflicts','nearest_noun','nearest_proper','nearest_pronoun',
         'same_sentence_noun','same_sentence_proper','possible_human_proper','possible_neuter_common','plural_agreement','candidate_previous_determiner')
CONTROL=POSITION+LEXICAL+PARTS+tuple('log_distance_'+f for f in ('noun','proper','pronoun','determiner'))


def sentence_ids(text,words):
    """Use only already-read punctuation between the previous word and the current word."""
    ids=[];sentence=0;previous=0
    for j,(a,b,*_) in enumerate(words):
        if j and re.search(r'[.!?]|[\r\n]',text[previous:a]):
            sentence+=1
        ids.append(sentence);previous=b
    return ids


def design(row,t,i):
    words=[row['text'][a:b] for a,b in row['words'][:t+1]];forms=[w.casefold() for w in words]
    tags=list(row['pos'][:t]);tags[-1]=row['pos_last'][i]
    freq=Counter(forms[:t]);last={w:j for j,w in enumerate(forms[:t])};query=FORMS[forms[t]]
    noun=np.asarray([k.startswith('NN') for k in tags]);proper=np.asarray([k in ('NNP','NNPS') for k in tags])
    pron=np.asarray([k.startswith('PRP') for k in tags]);det=np.asarray([k=='DT' for k in tags])
    distance=np.arange(t,0,-1);ld=np.log1p(distance);sd=np.array(row['sentences'][t])-np.array(row['sentences'][:t])
    if (sd<0).any():raise ValueError('Candidate is later than the current reading point')
    same=sd==0
    latest=[int(ix[-1]) if len(ix:=np.flatnonzero(m)) else -1 for m in (noun,proper,pron)]
    positions=np.arange(t)
    blocks=[ld,1/distance,np.exp(-distance/5),np.exp(-distance/20),np.exp(-distance/80),np.log1p(sd),same,
            [math.log1p(len(w)) for w in words[:t]],[w[0].isupper() for w in words[:t]],[w.isupper() for w in words[:t]],
            [w.isdecimal() for w in words[:t]],[math.log1p(freq[w]) for w in forms[:t]],
            [last[w]==j for j,w in enumerate(forms[:t])],[w==forms[t] for w in forms[:t]],
            [w in FORMS for w in forms[:t]],[FORMS.get(w)==query for w in forms[:t]],
            [w in FORMS and FORMS[w]!=query for w in forms[:t]],
            *[positions==k for k in latest],noun&same,proper&same,proper&(query in ('female','male')),
            noun&~proper&(query=='neuter'),[(t0 in ('NNS','NNPS'))==(query=='plural') for t0 in tags],
            np.r_[False,det[:-1]]]
    mapped=np.array([k if k in PARTS else 'other' for k in tags])
    blocks.extend([mapped==k for k in PARTS]);blocks.extend([ld*m for m in (noun,proper,pron,det)])
    result=np.column_stack(blocks).astype(np.float32)
    if result.shape!=(t,len(CONTROL)) or not np.isfinite(result).all():raise ValueError('Causal lexical pointer is malformed')
    return result
