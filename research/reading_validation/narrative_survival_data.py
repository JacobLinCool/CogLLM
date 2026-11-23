"""Construct censoring intervals and training-only semantic descriptions from author workbooks."""

from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b
import json
import re

import numpy as np

from research.reading_validation.memory_data import sha
from research.reading_validation.narrative_course_data import OUT, original_path
from research.reading_validation.narrative_course_features import unit_vector
from research.semantic_tracking.resolution_prepare import read_sheet


@dataclass(frozen=True)
class Spell:
    team: str
    question: int
    source: int
    definite: int
    upper: int | None
    uncertain_start: bool


def spells(story):
    all_spells=[]
    for team,states in story.teams.items():
        for q,x in enumerate(states.T):
            # Split only at a definite absence. A provisional state cannot create a new episode.
            for chunk in np.split(np.arange(len(x)),np.flatnonzero(x==0)):
                active=chunk[x[chunk]!=0]
                known=active[x[active]==1]
                if not len(known):
                    continue
                start,stop=map(int,(known[0],known[-1]))
                upper=int(active[-1]+1) if active[-1]+1<len(x) else None
                if upper is not None and (x[upper]!=0 or not np.all(x[stop+1:upper]==2)):
                    raise RuntimeError("Invalid interval inferred from source question states")
                all_spells.append(Spell(team,q,start,stop,upper,int(active[0])<start))
    return all_spells


def teacher(text,width=384):
    words=re.findall(r"[^\W\d_]+",text.casefold())
    feats=["word."+s for s in words]
    feats+=["pair."+a+" "+b for a,b in zip(words,words[1:])]
    for word in words:
        if len(word)>=4:
            s="^"+word+"$"
            feats.extend("trigram."+s[i:i+3] for i in range(len(s)-2))
    v=np.zeros(width)
    for term,count in Counter(feats).items():
        val=int.from_bytes(blake2b(term.encode(),digest_size=8).digest(),"big")
        v[val%width]+=(1. if (val>>32)&1 else -1.)*(1+np.log(count))
    return unit_vector(v)


def source_descriptions(stories):
    """Call only on the training stories of the current fit."""
    locked=json.loads((OUT/"input-seal.json").read_bytes())
    answers={}
    for story in stories:
        for team,state in story.teams.items():
            name,path=original_path(story.name,team)
            if sha(path)!=locked["files"][name]:
                raise ValueError("Wordings differ from the certified original source")
            rows=read_sheet(path)
            for col,v in rows[1].items():
                if col=="A" or not v.isdecimal():
                    continue
                q=int(v)-1
                if not rows[2].get(col,"").strip() or q>=state.shape[1]:
                    raise ValueError("Missing semantic description for an observed question")
                answers[story.name,team,q]=teacher(rows[2][col])
    return answers


def geometry(story):
    ss=spells(story)
    pairs=Counter((s.team,s.question) for s in ss)
    return dict(spells=len(ss),questions=len(pairs),additional_spells=sum(c-1 for c in pairs.values()),
                left_uncertain=sum(s.uncertain_start for s in ss),right_censored=sum(s.upper is None for s in ss),
                exact=sum(s.upper==s.definite+1 for s in ss),
                interval=sum(s.upper is not None and s.upper>s.definite+1 for s in ss))
