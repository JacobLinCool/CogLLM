"""Predict the visible boundaries of the earlier phrase surrounding a fixed pointer word."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


TAGS=("DT","PRP$","NN","NNS","NNP","NNPS","JJ","JJR","JJS","IN","CC","CD","PRP","POS","RB","VBN","other")
FUNCTIONS=("of","the","a","an","and","or","with","to","in","for","who","whose","that")
LENGTH=("log_distance","inverse_distance","at_seed","one_past","two_past","three_past")
NAMES=LENGTH+tuple(position+"_"+tag for position in ("seed","inside","neighbor") for tag in TAGS)+tuple(
    position+"_"+word for position in ("inside","neighbor") for word in FUNCTIONS)+(
    "adjacent_proper","adjacent_nouns","neighbor_determiner","inside_determiner","same_form","neighbor_title",
    "inside_title","neighbor_uppercase","inside_uppercase","neighbor_possessive","inside_possessive","punctuation",
    "comma","dash","semicolon","quote","sentence_boundary","number")
MAX_WORDS=34


@dataclass
class PriorWords:
    words:list
    tags:list
    gaps:list

    @classmethod
    def from_prefix(cls,text,bounds,tags,q):
        return cls([text[a:b] for a,b in bounds[:q]],list(tags[:q]),
                   [text[bounds[j][1]:bounds[j+1][0]] for j in range(q-1)])

    def edge(self,seed,j,step):
        k=j+step;g=self.gaps[min(j,k)];distance=abs(j-seed)
        left,right=self.words[j],self.words[k];a,b=left.casefold(),right.casefold()
        t,u=self.tags[j],self.tags[k]
        base=[np.log1p(distance),1/(distance+1),distance==0,distance==1,distance==2,distance==3]
        for code in (self.tags[seed],t,u):
            label=code if code in TAGS else "other";base.extend(label==v for v in TAGS)
        for v in (a,b):base.extend(v==w for w in FUNCTIONS)
        base.extend((t.startswith("NNP") and u.startswith("NNP"),t.startswith("NN") and u.startswith("NN"),
                     u in ("DT","PRP$"),t in ("DT","PRP$"),a==b,right.istitle(),left.istitle(),right.isupper(),
                     left.isupper(),b.endswith(("'s","’s")),a.endswith(("'s","’s")),bool(g.strip()),"," in g,
                     any(x in g for x in ("-","—","–")),";" in g,any(x in g for x in "'\"‘’“”"),
                     any(x in g for x in ".!?\n"),u=="CD"))
        result=np.asarray(base,dtype=float)
        if len(result)!=len(NAMES):raise ValueError("Original phrase boundary feature width changed")
        return result

    def chunk(self,seed):
        """Take the maximal local English noun chunk under a fixed, nonoverlapping POS grammar."""
        n=len(self.words);i=0
        while i<n:
            start=i
            if self.tags[i] in ("DT","PRP$"):i+=1
            while i<n and self.tags[i].startswith("JJ") and (i==start or not self.gaps[i-1].strip()):i+=1
            first=i
            while i<n and self.tags[i].startswith("NN") and (i==start or not self.gaps[i-1].strip()):i+=1
            if i>first:
                if start<=seed<i:return start,i
            else:i=start+1
        return seed,seed+1


@dataclass(frozen=True)
class PhraseInference:
    coefficients:dict
    width:int=MAX_WORDS

    @classmethod
    def load(cls,path,signature):
        from hashlib import sha256
        raw=Path(path).read_bytes()
        if sha256(raw).hexdigest()!=signature:raise ValueError("Reader phrase readout changed")
        d=json.loads(raw)
        if d["schema"]!="cogllm-english-antecedent-phrase-v1" or d["features"]!=list(NAMES) or d["width"]!=MAX_WORDS:
            raise ValueError("Unknown earlier-phrase contract")
        weights={}
        for name in ("left","right"):
            p=d["directions"][name];v=np.asarray(p["coef"],dtype=float)
            if v.shape!=(len(NAMES),) or not np.isfinite(v).all() or not np.isfinite(p["bias"]):raise ValueError("Invalid phrase hazard")
            weights[name]=(v,float(p["bias"]))
        return cls(weights,d["width"])

    def region(self,words,seed):
        distributions=[]
        for step,side in ((-1,"left"),(1,"right")):
            reach=min(self.width-1,seed if step<0 else len(words.words)-1-seed)
            if reach==0:distributions.append(np.array([0.]));continue
            beta,bias=self.coefficients[side]
            x=np.stack([words.edge(seed,seed+step*i,step)[:len(beta)] for i in range(reach)])
            z=x@beta+bias
            survive=-np.logaddexp(0.,-z);stop=-np.logaddexp(0.,z)
            lp=np.r_[0.,np.cumsum(survive)]
            # The final position is physically forced only when it is the actual prefix edge.
            if reach<self.width-1 or reach==(seed if step<0 else len(words.words)-1-seed):lp[:-1]+=stop
            else:
                next_edge=words.edge(seed,seed+step*reach,step)[:len(beta)]
                lp+=np.r_[stop,-np.logaddexp(0.,next_edge@beta+bias)]
            distributions.append(lp)
        l,r=distributions;total=l[:,None]+r[None,:]
        total[np.indices(total.shape).sum(0)>=self.width]=-np.inf
        a,b=np.unravel_index(total.argmax(),total.shape)
        return int(seed-a),int(seed+b+1)


PHRASE_MODEL_SHA="04d35d8a4a885db057134497315b03cdc60ca27587294cd7ec35709db1f15e79"
PHRASE_MODEL=Path(__file__).resolve().parents[1]/"results/reading-validation-transition-pointer/phrases/source-boundary-model.json"


def expand_links(text,pointer,links,model):
    """Attach independently learned phrase boundaries to existing word links without a second encoder call."""
    bounds=[tuple(w[:2]) for w in pointer.words]
    index={v:i for i,v in enumerate(bounds)}
    if len(index)!=len(bounds) or any(not 0<=a<b<=len(text) for a,b in bounds):raise ValueError("Invalid visible article word offsets")
    forms=[text[a:b] for a,b in bounds];out=[]
    for link in links:
        q=index[(link["pronoun_start"],link["pronoun_end"])];head=index[(link["antecedent_start"],link["antecedent_end"])]
        if q not in pointer.queries or head>=q:raise ValueError("Location and earlier-word index differ")
        tags=[p for _,p in pointer.model.tagger.tag(forms[:q+1])]
        words=PriorWords.from_prefix(text,bounds,tags,q);a,b=model.region(words,head)
        left,right=bounds[a][0],bounds[b-1][1]
        if not left<=bounds[head][0]<bounds[head][1]<=right<=bounds[q][0]:raise ValueError("Predicted phrase leaves the certified prior region")
        out.append({**link,"phrase_start":left,"phrase_end":right})
    return out
