"""Factor causal line representations into visible opening, resolution and source-pair terms."""

import re

import numpy as np

from research.reading_validation.narrative_course_features import unit_vector
from research.reading_validation.narrative_survival_data import spells,source_descriptions

GERMAN_FUNCTION=set("der die das den dem des und oder aber doch da zu dann so ein eine einen eines einer einem von mit "
                    "auf wie an in um ob er sie es ich du wir ihr nicht auch als war ist sind sich dass".split())


def pooled_types(text):
    tokens=re.findall(r"[^\W\d_]+",text.casefold())
    return {v for v in tokens if len(v)>2 and v not in GERMAN_FUNCTION}


class Lines:
    def __init__(self,story,native):
        self.story=story
        self.native=unit_vector(native)
        self.records=spells(story)
        texts=story.line_texts
        sets=[pooled_types(t) for t in texts]
        form=[]
        seen=set()
        for t,words in zip(texts,sets,strict=True):
            form.append([np.log1p(len(t)),np.log1p(len(words)),len(words-seen)/max(len(words),1),
                         float("?" in t),float("!" in t),any(c in t for c in "„“»«\""),
                         len(re.findall(r"\b[A-ZÄÖÜ][a-zäöüß]+",t)) / max(len(t.split()),1)])
            seen.update(words)
        self.form=np.array(form,dtype=float)
        self.n=story.n
        self.age=[];self.lex=[];self.semantic=[]
        self.ind=np.full((self.n,self.n),-1,dtype=int)
        self.blocks=[]
        previous_affinity=np.zeros(self.n)
        for t in range(1,self.n):
            earlier=np.arange(t)
            age=np.log1p(t-earlier)
            aa=np.column_stack([age,*[np.maximum(0,age-np.log1p(k)) for k in (2,8,32)]])
            same=np.array([len(sets[t]&sets[s]) for s in earlier])
            total=np.array([max(len(sets[t]|sets[s]),1) for s in earlier])
            lex=np.column_stack([self.form[earlier],np.repeat(self.form[t][None,:],t,axis=0),
                                 np.log1p(same),same/total])
            sims=np.column_stack([unit_vector(z[:t])@unit_vector(z[t]) for z in np.split(native,3,axis=1)])
            # Compare the current line to this candidate's history of semantic recurrence.
            prior=previous_affinity[:t].copy()
            previous_affinity[:t]=.85*prior+.15*sims[:,1]
            signal=np.log1p(100*story.relations[t,:t])
            sem=np.column_stack((sims,sims[:,1]-prior,prior,signal))
            self.age.append(aa);self.lex.append(lex);self.semantic.append(sem)
            start=t*(t-1)//2
            self.ind[t,:t]=start+earlier
            self.blocks.append(slice(start,start+t))
        self.age=np.vstack(self.age);self.lex=np.vstack(self.lex);self.semantic=np.vstack(self.semantic)
        # For the opening, use the current line and a causal decaying summary of the setting.
        avg=np.zeros_like(self.native)
        for t in range(1,self.n):
            avg[t]=.75*avg[t-1]+.25*self.native[t-1]
        self.origin=np.column_stack((self.native,avg))
        self.cue=np.column_stack((self.native,self.native-avg))
        self.birth_form=np.column_stack((np.log1p(np.arange(self.n)),self.form))


def ridge_map(x,y,w):
    """Source-weighted shrinkage in the dual, which is small for four annotated stories."""
    w=w/w.sum()
    mx=w@x;my=w@y
    xc=x-mx;yc=y-my
    # Input row normalization fixes the prior scale across opening and closing maps.
    xx=unit_vector(xc);a=xx*np.sqrt(w)[:,None]
    lhs=a@a.T+.02*np.eye(len(w))
    rhs=yc*np.sqrt(w)[:,None]
    coef=np.linalg.solve(lhs,rhs)
    if max(np.abs(lhs@coef-rhs).flat)>1e-8:
        raise ArithmeticError("Question-word regression failed its source normal equation")
    return mx,my,a.T@coef


def learn_teacher(pages):
    questions=source_descriptions([p.story for p in pages])
    models=[]
    for role in ("origin","cue"):
        xs=[];ys=[];ws=[]
        for p in pages:
            candidates=[r for r in p.records if role=="origin" or r.upper==r.definite+1]
            teams={r.team:sum(v.team==r.team for v in candidates) for r in candidates}
            for r in candidates:
                t=r.source if role=="origin" else r.upper
                xs.append(getattr(p,role)[t]);ys.append(questions[p.story.name,r.team,r.question])
                ws.append(1/(len(pages)*len(teams)*teams[r.team]))
        models.append(ridge_map(np.stack(xs),np.stack(ys),np.array(ws)))
    return models


def wording_pair(page,models):
    vectors=[]
    for role,(mx,my,b) in zip(("origin","cue"),models,strict=True):
        # Center the predicted wording so generic question words cannot dominate affinity.
        vectors.append(unit_vector(unit_vector(getattr(page,role)-mx)@b))
    opening,cue=vectors
    return np.concatenate([opening[:t]@cue[t] for t in range(1,page.n)])[:,None]


def pair_design(page,arm,teacher=None):
    parts=[page.age]
    if arm!="duration":
        parts.append(page.lex)
    if arm in ("native","wording"):
        parts.append(page.semantic)
    if arm=="wording":
        parts.append(wording_pair(page,teacher))
    return np.column_stack(parts)
