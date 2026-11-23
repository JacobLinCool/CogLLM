"""Quantify held discrimination after holding source and summarizer identity fixed."""
from collections import defaultdict
from itertools import combinations
import numpy as np
from research.reading_validation.transition_seahorse_score import summary
from research.reading_validation.transition_seahorse_fit import key, corpus, LABEL
from research.reading_validation.transition_seahorse_data import DEST, source_rows, sha
from research.reading_validation.transition_seahorse_features import load,save

METHODS=("frozen_form_raw","frozen_native_raw","form","native","metadata","metadata_form","metadata_native","character","character_form","character_native")


def grouped_rank(y,p,ix):
    order=ix[np.argsort(p[ix],kind="stable")]
    s=p[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    return order, starts, y[order]


def auc_counter(included,order,starts,y):
    mass=included[:,order]
    pos=np.add.reduceat(mass*y,starts,axis=1)
    neg=np.add.reduceat(mass*(1-y),starts,axis=1)
    wins=(pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(1)
    totals=pos.sum(1)*neg.sum(1)
    return wins,totals


def grouped_auc(y,values,raters,ctrl,C):
    eligible=[v for v in ctrl if len(np.unique(y[v]))==2]
    own=[grouped_rank(y,values,ix) for ix in eligible]
    out=[]
    for l in range(0,len(C),50):
        w=C[l:l+50,raters].astype(float)
        sx=np.zeros(len(w));n=np.zeros(len(w))
        for info in own:
            a,b=auc_counter(w,*info);sx+=a;n+=b
        out.extend(np.divide(sx,n,where=n>0,out=np.full_like(sx,np.nan)))
    return np.array(out)


def capture():
    p=DEST/"forecasts-before-held-responses.npz"
    if sha(p)!=load(DEST/"fit-seal.json")["forecast_sha256"]:
        raise ValueError("Held predictions or operating conditions changed")
    rows={key(r):r for r in source_rows("test",grades=True)}
    with np.load(p,allow_pickle=False) as d:
        keep=np.array([rows[k]["question1"] in LABEL for k in d["keys"]])
        ids=d["keys"][keep];pred={k:d[k][keep] for k in METHODS}
    r=[rows[k] for k in ids]
    return r,pred


def main():
    r,pred=capture();y=np.array([LABEL[d["question1"]] for d in r])
    s=np.array([d["gem_id"] for d in r]);g,gid=np.unique(s,return_inverse=True)
    system=np.array([d["model"] for d in r]);domain=np.array([corpus(d["gem_id"]) for d in r])
    types=np.char.add(np.char.add(domain,"/"),system)
    controls=[np.flatnonzero(types==t) for t in np.unique(types)]
    rng=np.random.default_rng(97114);C=rng.multinomial(len(g),np.full(len(g),1/len(g)),size=2500).astype(np.int16)
    C=np.vstack((np.ones(len(g),dtype=np.int16),C))
    ranks={}; raw={}
    for name,p in pred.items():
        um={}
        for label,groups in [("all",[np.arange(len(y))]),("within_corpus_and_generator",controls)]:
            z=grouped_auc(y,p,gid,groups,C)
            um[label]=summary(z[0],z[1:]);raw[name,label]=z
        ranks[name]=um
    differences={}
    for a,b in (("native","form"),("metadata_native","metadata_form"),("character_native","character_form"),("native","metadata"),("frozen_native_raw","frozen_form_raw")):
        for name in ("all","within_corpus_and_generator"):
            z=raw[a,name]-raw[b,name]
            differences[a+" minus "+b+" "+name]=summary(z[0],z[1:])
    # Within one corpus and fixed generator pair, use only article reversals in the binary judgments.
    pools=defaultdict(lambda:defaultdict(list))
    for j,story in enumerate(g):
        obs=np.flatnonzero(gid==j)
        for a,b in combinations(obs,2):
            if y[a]==y[b] or system[a]==system[b]:continue
            if system[a]>system[b]:a,b=b,a
            pools[domain[a],system[a],system[b]][int(y[a]>y[b])].append((a,b,j))
    packed=[]
    for roles in pools.values():
        if not roles[0] or not roles[1]:continue
        for a,b,x in roles[1]:
            for c,d,z in roles[0]:
                if x==z:raise ValueError("Inconsistent reversal within a source story")
                packed.append((a,b,c,d,x,z))
    matrix=np.array(packed,dtype=int)
    a,b,c,d,gs,gt=matrix.T
    suite={}; sens={}
    for method,p in pred.items():
        delta=(p[a]-p[b])-(p[c]-p[d])
        ok=(delta>0)+.5*(delta==0);tmp=[]
        for left in range(0,len(C),20):
            weight=C[left:left+20,gs].astype(float)*C[left:left+20,gt]
            tmp.extend((weight@ok)/weight.sum(1))
        z=np.array(tmp);sens[method]=z;suite[method]=summary(z[0],z[1:])
    pairdiff={}
    for a,b in (("native","form"),("native","metadata"),("character_native","character_form")):
        z=sens[a]-sens[b];pairdiff[a+" minus "+b]=summary(z[0],z[1:])
    save(DEST/"ranking-sensitivity.json",{"scope":"Pooled exact AUROC conditions and post-primary producer/story reversal sensitivity",
        "source":sha(DEST/"forecasts-before-held-responses.npz"),"article_draws":len(C)-1,
        "held":len(r),"articles":len(g),"within_generator_and_corpus":{"informative_strata":sum(len(np.unique(y[i]))==2 for i in controls),
             "strata":len(controls),"opposite_verdict_pairs":sum(int(y[i].sum())*(len(i)-int(y[i].sum())) for i in controls)},
        "ranks":ranks,"paired":differences,"crossovers":{"contrasts":len(packed),"generator_pairs":len([1 for v in pools.values() if v[0] and v[1]]),
             "different_articles":len(set(zip(gs.tolist(),gt.tolist()))),"scores":suite,"paired":pairdiff}})
    print("Conditional discrimination",ranks["native"],differences["native minus form within_corpus_and_generator"],"reversals",len(packed),suite["native"],suite["form"],pairdiff,flush=True)


if __name__=="__main__":
    main()
