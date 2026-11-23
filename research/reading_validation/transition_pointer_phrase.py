"""Fit phrase-boundary continuation decisions on prior choices and check untouched English corpora."""

from argparse import ArgumentParser
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.backbone_transfer.extract import digest as sha,write_json as dump
from research.reading_validation.antecedent_phrase_inference import LENGTH,MAX_WORDS,NAMES,PhraseInference,PriorWords


ROOT=Path(".cache/cogllm/reading-validation")
SRC=Path("research/results/reading-validation-transition-pointer")
GAP=Path("research/results/reading-validation-transition-pointer-gap")
OUT=SRC/"phrases"
PR=Path("research/reading_validation/transition-pointer-phrase-protocol.md")
METHODS=("one_word","noun_chunk","length","phrase")
METRICS=("exact","iou","token_precision","token_recall","predicted_words","observed_words","anchor_inside")


def derive(row,q):
    i=row["queries"].index(q);tags=list(row["pos"][:q]);tags[-1]=row["pos_last"][i]
    return PriorWords.from_prefix(row["text"],row["words"],tags,q)


def examples(source):
    by=Counter();arrays={"left":[],"right":[]}
    for r in source:
        if r["fold"]=="train":by[r["group"]]+=len(r["queries"])
    widths=Counter()
    for r in source:
        if r["fold"]!="train":continue
        for q in r["queries"]:
            words=derive(r,q);votes=r["clicks"][str(q)];n=sum(c for _,_,c,*_ in votes)
            norm=by[r["group"]]*len(by)*n
            for a,b,c,*_ in votes:
                widths[b-a]+=c
                for seed in range(a,b):
                    w=c/(norm*(b-a))
                    for direction,step in (("left",-1),("right",1)):
                        j=seed
                        while 0<=j+step<q:
                            t=a<=j+step<b
                            arrays[direction].append((words.edge(seed,j,step),int(t),w))
                            if not t:break
                            j+=step
    if max(widths)!=MAX_WORDS or sum(widths.values())!=8283:raise ValueError("There is a change in the observed source phrase support")
    return arrays,widths


def fit(x,y,mass):
    w=mass/mass.sum();mu=w@x
    sd=np.sqrt(w@((x-mu)**2));sd[sd<1e-6]=1.;z=(x-mu)/sd
    def obj(t):
        eta=z@t[:-1]+t[-1];p=expit(eta)
        diff=w*(p-y);pen=.005*np.sum(t[:-1]**2)
        return w@(np.logaddexp(0.,eta)-y*eta)+pen,np.r_[z.T@diff+.01*t[:-1],diff.sum()]
    opt=minimize(obj,np.zeros(z.shape[1]+1),jac=True,method="L-BFGS-B",options={"ftol":1e-14,"gtol":2e-9,"maxiter":400})
    grad=float(np.abs(opt.jac).max())
    if grad>1e-6 or not np.isfinite(grad+opt.fun):raise ValueError("Source boundary hazard failed stationarity")
    raw=opt.x[:-1]/sd;bias=float(opt.x[-1]-mu@raw)
    return (raw,bias),{"events":len(y),"positive_extensions":int(y.sum()),"gradient":grad,"steps":int(opt.nit),"objective":float(opt.fun)}


def evaluate(region,click,seed):
    x,y=region;a,b=click
    overlap=max(0,min(b,y)-max(a,x));width=y-x;true=b-a
    return np.array([float(region==click),overlap/(width+true-overlap),overlap/width,overlap/true,width,true,float(a<=seed<b)])


def collect(table,group,r,q,original,logits,models):
    z=logits.astype(float);seed=int(z.argmax());confidence=float(1/np.exp(z-z.max()).sum());words=derive(r,q)
    guesses=[(seed,seed+1),words.chunk(seed),models["length"].region(words,seed),models["phrase"].region(words,seed)]
    for subset,clicks in (("all",original),("multiword",[c for c in original if c[1]-c[0]>1]),
                          ("single_word",[c for c in original if c[1]-c[0]==1]),
                          ("source_confidence_at_least_80",original if confidence>=.8 else [])):
        if not clicks:continue
        n=sum(c[2] for c in clicks);m=np.zeros((len(METHODS),len(METRICS)))
        for a,b,v,*_ in clicks:
            for j,pred in enumerate(guesses):m[j]+=evaluate(pred,(a,b),seed)*v/n
        table[subset][group].append((m,n))
    return guesses[-1]


def scores(groups):
    keys=sorted(groups);matrix=np.stack([np.mean([m for m,_ in groups[k]],axis=0) for k in keys])
    pt=matrix.mean(0);rng=np.random.default_rng(3445)
    boot=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=5000)@matrix.reshape(len(keys),-1)/len(keys)
    boot=boot.reshape(-1,*pt.shape)
    ci=lambda a,v:[float(a),*np.quantile(v,[.025,.975]).tolist()]
    methods={name:{k:ci(pt[j,i],boot[:,j,i]) for i,k in enumerate(METRICS)} for j,name in enumerate(METHODS)}
    paired={"phrase_minus_"+name:{k:ci(pt[-1,i]-pt[j,i],boot[:,-1,i]-boot[:,j,i]) for i,k in enumerate(METRICS)} for j,name in enumerate(METHODS[:-1])}
    return {"article_groups":len(keys),"queries":sum(len(v) for v in groups.values()),"reader_decisions":sum(n for v in groups.values() for _,n in v),
            "methods":methods,"paired":paired}


def main():
    source_file=ROOT/"pointer-stimuli.jsonl";source_seal=json.loads((SRC/"source-seal.json").read_text())
    gpacket=ROOT/"gap-pointer-stimuli.jsonl";gs=json.loads((GAP/"source-seal.json").read_text())
    if sha(source_file)!=source_seal["input_sha256"] or sha(gpacket)!=gs["input_sha256"]:raise ValueError("Source stimulus changed")
    source=list(map(json.loads,source_file.read_text().splitlines()))
    data,widths=examples(source);wts={k:{} for k in ("length","phrase")};diagnostics={}
    for side,v in data.items():
        x=np.stack([r[0] for r in v]);y,mass=np.array([(r[1],r[2]) for r in v]).T
        for mode,size in (("length",len(LENGTH)),("phrase",len(NAMES))):
            wts[mode][side],diag=fit(x[:,:size],y,mass);diagnostics[mode+"_"+side]=diag
    OUT.mkdir(parents=True,exist_ok=True)
    models={k:PhraseInference(t) for k,t in wts.items()}
    full=OUT/"source-boundary-model.json"
    dump(full,{"schema":"cogllm-english-antecedent-phrase-v1","protocol_sha256":sha(PR),"source_sha256":sha(source_file),
               "features":NAMES,"width":MAX_WORDS,"directions":{k:{"coef":v.tolist(),"bias":b} for k,(v,b) in wts["phrase"].items()}})
    models["phrase"]=PhraseInference.load(full,sha(full))
    stats={};outputs={}
    for name,path,packet,expected in (("game",SRC,source_file,source_seal),("gap",GAP,gpacket,gs)):
        keypath=path/("external-clicks.json" if name=="game" else "external-regions.json")
        if sha(keypath)!=expected["gold_sha256"]:raise ValueError("External human phrase key changed")
        receipt=json.loads((path/"receipt.json").read_text());pf=path/"candidate-word-logits.npz"
        if sha(pf)!=receipt["predictions_sha256"]:raise ValueError("Frozen word locations changed")
        article={r["id"]:r for r in map(json.loads,packet.read_text().splitlines())}
        annotation=json.loads(keypath.read_text());measurements=[]
        if name=="game":
            for d,qs in annotation["articles"].items():
                for q,v in qs.items():measurements.append((d,d,int(q),v["spans"]))
        else:
            for r in annotation["rows"]:
                region=r["B"] if r["target"] else r["A"];measurements.append((r["article"],r["id"],r["q"],[[min(region),max(region)+1,1]]))
        table=defaultdict(lambda:defaultdict(list)); outputs[name]={}
        with np.load(pf) as values:
            for group,d,q,choices in measurements:
                reg=collect(table,group,article[d],q,choices,values[d+"/"+str(q)][-1],models)
                outputs[name][d+"/"+str(q)]=list(reg)
        stats[name]={k:scores(v) for k,v in table.items()}
    approved=all(stats[name]["all"]["paired"]["phrase_minus_one_word"]["iou"][1]>0 and
                 stats[name]["all"]["paired"]["phrase_minus_noun_chunk"]["iou"][2]>=0 for name in ("game","gap"))
    dump(OUT/"results.json",{"protocol_sha256":sha(PR),"source_hazard_sha256":sha(full),"source_fit_diagnostics":diagnostics,
         "source_click_widths":dict(sorted(widths.items())),"qualified_for_optional_phrase_display":approved,**stats})
    dump(OUT/"heldout-phrases.json",outputs)
    print(approved,{dataset:{k:v["iou"] for k,v in ds["all"]["methods"].items()} for dataset,ds in stats.items()},flush=True)


if __name__=="__main__":main()
