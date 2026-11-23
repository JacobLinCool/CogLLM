"""Seal the public Dutch carrier sentences and a separately stored, outcome-free meaning vocabulary."""
import csv
from collections import Counter
from hashlib import sha256
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linear_sum_assignment
from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.identity_prepare import DATA,TOKENIZER,sealed_rows

ROOT=DATA/'exposure'
OUT=Path('research/results/reading-validation-exposure')
PROTOCOL=Path(__file__).with_name('exposure-protocol.md')


def read_csv(path):
    with path.open(encoding='cp1252',newline='') as f:return list(csv.DictReader(f))


def key(text):
    return sha256(text.encode()).hexdigest()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);tok=Tokenizer.from_file(str(TOKENIZER))
    words=sorted(read_csv(ROOT/'stimuli_exp2.csv'),key=lambda r:r['Word']);cues=[];onsets=[];wds=[]
    for r in words:
        short=r['Retrieval'].strip();rich=r['Context'].strip()
        if not rich.endswith(short):raise ValueError('Changed carrier across randomized conditions')
        cues.append(rich[:-len(short)].strip());wds.append(r['Translation'].strip().casefold())
        onsets.append(tok.encode(' '+wds[-1],add_special_tokens=False).ids[0])
    lengths=np.array([len(tok.encode(s,add_special_tokens=False).ids) for s in cues])
    mask=np.equal.outer(cues,cues);dist=np.abs(lengths[:,None]-lengths[None,:])
    # Small deterministic noise resolves equivalent matches without using any annotations.
    cost=dist.astype('float64')+np.random.default_rng(4712).random(dist.shape)/1000;cost[mask]=1e8
    ix,perm=linear_sum_assignment(cost)
    if np.any(mask[ix,perm]) or not np.array_equal(ix,np.arange(len(words))):raise ValueError('Failed derangement')
    stimuli={};lookup=[];excluded=Counter()
    for j,r in enumerate(words):
        short=r['Retrieval'].strip();rich=r['Context'].strip();late=r['Retrieval+Context'].strip();foil=cues[perm[j]]+' '+short
        if not late.startswith(short) or late[len(short):].strip()!=cues[j]:
            raise ValueError('The late support was not the same cue and carrier')
        local={}
        for kind,s in {'sparse':short,'before':rich,'after':late,'foil':foil}.items():
            # Only the carrier mention is relevant; a distractor may contain similar characters.
            off=0 if kind in ('sparse','after') else len(s)-len(short)
            matches=list(re.finditer(re.escape(r['Word']),short,re.I))
            if len(matches)!=1:raise ValueError('Ambiguous experimental carrier mention')
            a,b=[off+v for v in matches[0].span()];enc=tok.encode(s,add_special_tokens=False)
            ids=[i for i,(l,h) in enumerate(enc.offsets) if h>a and l<b and h>l]
            if not ids or ids[0]==0:excluded['first_or_unmapped_token']+=1;local={};break
            prefix_end=enc.offsets[ids[0]][0]
            if tok.encode(s[:prefix_end],add_special_tokens=False).ids!=enc.ids[:ids[0]]:raise ValueError('Unstable pre-noun native prefix')
            uid=key(s);record={'id':uid,'text':s};stimuli[uid]=record
            local[kind]={'id':uid,'end':ids[0]-1,'chars':[a,b]}
        if local:
            lookup.append({'word':r['Word'],'meaning':wds[j],'onset':onsets[j],'arms':local,
                           'cue_tokens':int(lengths[j]),'foil_tokens':int(lengths[perm[j]]),'unique_onset':onsets.count(onsets[j])==1})
    codes=sorted(set(onsets));write_json(ROOT/'vocabulary.json',{'onsets':codes})
    used={arm['id'] for row in lookup for arm in row['arms'].values()};sealed_rows(ROOT/'stimuli.jsonl',[stimuli[k] for k in sorted(used)])
    write_json(ROOT/'items.json',lookup)
    write_json(OUT/'input-seal.json',{'protocol':digest(PROTOCOL),'stimuli':digest(ROOT/'stimuli.jsonl'),'vocabulary':digest(ROOT/'vocabulary.json'),
        'mapping':digest(ROOT/'items.json'),'source_stimuli':{p.name:digest(p) for p in sorted(ROOT.glob('stimuli_exp*.csv'))},
        'source_outcome_hashes':{p.name:digest(p) for p in sorted(ROOT.glob('Experiment*.csv'))},'excluded':dict(excluded),'eligible':len(lookup),
        'unique_texts':len(used),'distinct_meaning_onsets':len(codes),'unique_onset_items':sum(r['unique_onset'] for r in lookup),
        'no_trial_outcomes_read':True,'prior_access':'Article-level average effects, source practice conditions, item fields and schema previously inspected.'})


if __name__=='__main__':prepare()
