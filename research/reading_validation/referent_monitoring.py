"""Learn referent selection and human first-report hazards without access to the held family."""

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp,softmax

from research.reading_validation.expectation_data import OUT as EXPECT,WORK
from research.reading_validation.memory_data import BASE,E1,OUT as PRIOR,load_rows,sha
from research.reading_validation.memory_fit import counts
from research.reading_validation.memory_score import reveal
from research.reading_validation.referent_features import lexical_features,setup,source_carriers

OUT=Path("research/results/reading-validation-referent-monitoring")
PLAN=Path(__file__).with_name("referent-monitoring-protocol.md")
EST=Path(__file__).with_name("referent-estimator.md")


def location(items,docs):
    target=[Counter(),Counter()];background=[Counter(),Counter()]
    for r in items:
        for j,s in enumerate(docs[r["id"]],1):
            if j==1:
                continue
            bucket=target if j==r["target"] else background
            bucket[0][j]+=1;bucket[1][len(s)]+=1
    nt=sum(target[0].values());nb=sum(background[0].values())
    def predict(j,words):
        odds=nt/nb
        for i,key in enumerate((j,words)):
            possible=len(set(target[i])|set(background[i]))+1
            a=(target[i][key]+.5)/(nt+.5*possible)
            b=(background[i][key]+.5)/(nb+.5*possible)
            odds*=a/b
        return odds/(1+odds)
    return predict


def feature_matrix(doc,arm,t):
    a=doc["forms"][t];l=doc["lex"][t]
    if arm=="lexical":
        return np.column_stack((a,l))
    return np.column_stack((a,l,doc["neural"][t]))


def optimize(fn,init):
    opt=minimize(fn,init,jac=True,method="L-BFGS-B",options={"maxiter":600,"gtol":1e-8,"ftol":1e-14})
    if not opt.success or max(abs(opt.jac))>1e-5:
        raise ArithmeticError(f"Referent observation fit failed convergence: {opt.message}")
    return opt.x,float(max(abs(opt.jac)))


def selector(learning,features,carriers,arm):
    sets=[];gold=[]
    for r in learning:
        if r["study"]!=1 or r["story"] not in carriers:
            continue
        sets.append(feature_matrix(features[r["id"]],arm,r["target"]-1))
        gold.append(carriers[r["story"]])
    data=np.vstack(sets);mu=data.mean(0);sd=data.std(0);sd[sd<1e-6]=1
    sets=[(x-mu)/sd for x in sets]
    def objective(v):
        val=0.;g=np.zeros_like(v)
        for x,y in zip(sets,gold,strict=True):
            z=x@v
            val+=logsumexp(z)-z[y];g+=softmax(z)@x-x[y]
        return val/len(sets)+.01*(v@v),g/len(sets)+.02*v
    v,e=optimize(objective,np.zeros(len(mu)))
    return mu,sd,v,e


def present(model,doc,arm,t):
    x=feature_matrix(doc,arm,t);mu,sd,v,_=model
    p=softmax(((x-mu)/sd)@v)
    return p@x


def classify(learning,features,words,arm):
    x=np.stack([present(words,features[r["id"]],arm,r["target"]-1) for r in learning])
    classes=sorted({r["arm"] for r in learning})
    indices=[classes.index(r["arm"]) for r in learning]
    cases=Counter(r["study"] for r in learning)
    w=np.array([1/(len(cases)*cases[r["study"]]) for r in learning])
    mu=w@x;sd=np.sqrt(w@((x-mu)**2));sd[sd<1e-6]=1
    z=np.column_stack((np.ones(len(x)),(x-mu)/sd));y=np.eye(len(classes))[indices]
    shape=(z.shape[1],len(classes))
    def objective(raw):
        v=raw.reshape(shape);u=z@v
        loss=w@(logsumexp(u,axis=1)-(y*u).sum(1))+.01*np.square(v[1:]).sum()
        grad=z.T@(w[:,None]*(softmax(u,axis=1)-y));grad[1:]+=.02*v[1:]
        return loss,grad.ravel()
    v,e=optimize(objective,np.zeros(np.prod(shape)))
    return classes,mu,sd,v.reshape(shape),e


def read_arms(rows,docs,features,carriers,arm,cache,held_ids):
    # Each requested model is identified by its precise included families and allowed studies.
    mode=tuple(sorted((r["study"],r["story"],r["id"]) for r in rows))
    key=(mode,arm)
    if key not in cache:
        locator=location(rows,docs)
        if arm!="role":
            select=selector(rows,features,carriers,arm)
            cls=classify(rows,features,select,arm)
            errors=[select[-1],cls[-1]]
        else:
            select=cls=None;errors=[]
        cache[key]=(locator,select,cls,errors,{})
    locator,select,cls,errors,results=cache[key]
    for k in held_ids:
        if k in results:
            continue
        x=[]
        for t,sentence in enumerate(docs[k]):
            role=locator(t+1,len(sentence))
            if arm=="role":
                x.append([role])
            else:
                names,mu,sd,b,_=cls
                z=present(select,features[k],arm,t)
                prob=softmax(np.r_[1.,(z-mu)/sd]@b)
                decisions=dict(zip(names,prob,strict=True))
                x.append([role,role*decisions["inconsistent"],role*decisions.get("neutral",0.)])
        results[k]=np.array(x)
    return results,errors


def calibrate(rows,features,counts_):
    cases=Counter(r["study"] for r in rows);xx=[];yy=[];ww=[]
    for r in rows:
        k=r["id"];at,event,n=counts_[r["study"],k];risk=(at>0);risk[0]=False
        w=at[risk].astype(float);w/=w.sum()
        ix=np.flatnonzero(risk)==r["target"]-1
        if ix.sum()!=1:
            raise ValueError("Source target has no observed reading opportunity")
        w[ix]+=1;w/=2*len(cases)*cases[r["study"]]
        xx.append(features[k][risk]);yy.append(event[risk]/at[risk]);ww.append(w)
    x=np.vstack(xx);w=np.concatenate(ww);y=np.concatenate(yy)
    mu=w@x;sd=np.sqrt(w@((x-mu)**2));sd[sd<1e-7]=1
    x=np.column_stack((np.ones(len(x)),(x-mu)/sd))
    def objective(v):
        z=x@v
        loss=w@(np.logaddexp(0,z)-y*z)+.01*(v[1:]@v[1:])
        grad=x.T@(w*(expit(z)-y));grad[1:]+=.02*v[1:]
        return loss,grad
    v,error=optimize(objective,np.zeros(x.shape[1]))
    return mu,sd,v,error


def run(native,dest):
    if dest.exists():
        raise FileExistsError(dest)
    setup()
    rows=json.loads((BASE/"memory/study-map.json").read_bytes())
    docs={r["id"]:r["sentences"] for r in load_rows(BASE/"memory/stories.jsonl")}
    source=[r for r in rows if r["study"] in (1,2)]
    observed={}
    for study in (1,2):
        sample=[r for r in source if r["study"]==study]
        people=load_rows(E1/"outcomes.jsonl") if study==1 else reveal(2,rows)
        observed.update({(study,k):v for k,v in counts(people,sample).items()})
    carrier=source_carriers([r for r in source if r["study"]==1],docs)
    arrays={}
    if native:
        seal=json.loads((native/"receipt.json").read_bytes())
        if sha(native/"features.npz")!=seal["file"]:
            raise ValueError("Single-pass features have changed")
        remap=json.loads((WORK/"opportunities.json").read_bytes())["frick"]
        with np.load(native/"features.npz",allow_pickle=False) as f:
            arrays={k:(f[code+".word"],f[code+".features"]) for k,code in remap.items()}
    feature={k:lexical_features(sentences,arrays.get(k)) for k,sentences in docs.items()}
    tokens={k:[[words[a:b] for a,b in spans] for spans in feature[k]["spans"]]
            for k,ss in docs.items() for words in ["\n\n".join(ss)]}
    active={"role":"role","lexical":"lexical"}
    if native:
        active["expectation"]="expectation";active["e1_expectation"]="expectation"
    else:
        active["e1_lexical"]="lexical"
    predictions={};diagnostics={};cache={}
    with np.load(PRIOR/"forecast/forecasts.npz",allow_pickle=False) as old:
        for k in docs:
            predictions["strict.original."+k]=old["strict.frozen."+k]
    for fid in sorted({r["story"] for r in rows if r["study"]==3}):
        for name,arm in active.items():
            train=[r for r in source if r["story"]!=fid and (not name.startswith("e1_") or r["study"]==1)]
            validation={};worst=0.
            for g in sorted({r["story"] for r in train}):
                ids={r["id"] for r in train if r["story"]==g}
                learned,errors=read_arms([r for r in train if r["story"]!=g],tokens,feature,carrier,arm,cache,ids)
                validation.update({k:learned[k] for k in ids})
                worst=max([worst,*errors])
            mu,sd,v,err=calibrate(train,validation,observed)
            ids={r["id"] for r in rows if r["story"]==fid and r["study"]>=2}
            test,errors=read_arms(train,tokens,feature,carrier,arm,cache,ids)
            for k in ids:
                y=expit((test[k]-mu)/sd@v[1:]+v[0]);y[0]=0.
                predictions["strict."+name+"."+k]=y
            diagnostics[f"{fid}.{name}"]=dict(max_gradient=max([err,worst,*errors]),source_records=len(train),
                 source_studies=sorted({r["study"] for r in train}),source_families=sorted({r["story"] for r in train}),
                 held_family=fid,coef=v.tolist(),mu=mu.tolist(),sd=sd.tolist())
    dest.mkdir(parents=True)
    np.savez_compressed(dest/"forecasts.npz",**predictions)
    (dest/"fits.json").write_text(json.dumps(diagnostics,allow_nan=False)+"\n")
    (dest/"seal.json").write_text(json.dumps(dict(protocol=sha(PLAN),estimator=sha(EST),native=sha(native/"receipt.json") if native else None,
         wordnet=sha(BASE/"nltk/corpora/wordnet.zip"),source_carriers=len(carrier),experiment3_responses_in_fit=0,
         models=["original",*active],files={n:sha(dest/n) for n in ("forecasts.npz","fits.json")}),indent=2)+"\n")
    print("Learned",len(diagnostics),"source-blocked fits with",len(carrier),"unambiguous training word identities")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native",type=Path);p.add_argument("--dest",type=Path,required=True)
    run(**vars(p.parse_args()))
