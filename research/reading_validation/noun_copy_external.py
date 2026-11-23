"""Forecast each unseen work before grading people's separate noun-reference choices."""
from collections import Counter,defaultdict
from itertools import chain
from pathlib import Path
import argparse,json
import numpy as np
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.nominal_data import OUT,ROOT,REJECT,restore
from research.reading_validation.nominal_estimator import Words
from research.reading_validation.noun_copy import log_forecasts,model_weights,query
from research.reading_validation.transition_direct_data import gold_docs
from research.reading_validation.transition_game_prepare import silver_docs
from research.reading_validation.transition_pointer_score import hit_k

CACHE=ROOT/"extra-nominal-forecasts"
ARMS=("surface","context","fixed_copy","learned_copy")

def originals():
    return list(map(json.loads,(ROOT/"null-choice-pointer-stimuli.jsonl").read_text().splitlines()))

def forecast(native,previous):
    aud=json.loads((OUT/"extra-exposure.json").read_text());new=json.loads((native/"receipt.json").read_text())
    old=json.loads((previous/"receipt.json").read_text())
    if new["input_sha256"]!=aud["source"] or len(originals())!=487 or digest(native/"word-states.npz")!=new["archive_sha256"] or digest(previous/"word-states.npz")!=old["archive_sha256"]:raise ValueError("Archive or word geometry changed")
    mu,basis,models,copy=model_weights();files={};CACHE.mkdir(exist_ok=True);eligible=0;total=0;parity=[]
    with np.load(native/"word-states.npz") as target,np.load(previous/"word-states.npz") as control:
        for r in originals():
            d=r["id"];k=d+"__h"
            if k not in target.files:continue
            raw=target[k].astype(float)
            if r["fold"]=="dev":
                parity.append(float(np.max(np.abs(raw-control[k]))));continue
            if r["fold"]!="new":raise ValueError("Unexpected held exposure")
            doc=Words(r,(raw-mu)@basis);cuts=[];acts=[];logprobs=[]
            for q,w in enumerate(doc.forms):
                if not q or w in REJECT or not w[0].isalpha() or not w[-1].isalpha():continue
                a,p=query(doc,q,models,copy);acts.append(a);logprobs.append(p.astype("float32"));cuts.append(q)
            file=CACHE/(d+".npz")
            np.savez_compressed(file,q=np.array(cuts),act=np.stack(acts) if acts else np.empty((0,4,4)),
                                words=np.concatenate(logprobs,axis=-1) if logprobs else np.empty((4,0)))
            files[file.name]=digest(file);eligible+=len(acts);total+=sum(cuts)
    if len(files)!=201 or len(parity)!=2 or max(parity)>.03:raise ValueError(f"New/old original-word parity failed: {len(files)}, {parity}")
    write_json(OUT/"copy-external-forecast.json",{"gate":digest(OUT/"copy-fit.json"),"raw":new,"previous_archive":old["archive_sha256"],
                "word_sketch_parity":parity,"words":eligible,"candidate_words":total,"files":files,"target_outcomes_used":0})

def reveal():
    raw={r["id"]:r for r in originals()};seen=set()
    for d,g,f,tr,cr,el,ts in silver_docs():
        if d in raw and raw[d]["fold"] in ("train","dev"):seen.update(a.get("user_id") for a in tr.iter("anchor") if a.get("mode")=="a")
    all_obs={};fresh={};counts=Counter();people=set();held_people=set()
    for d,g,f,tr,cr,el,ts in chain(silver_docs(),((d,g,"test",tr,cr,el,ts) for d,g,tr,cr,el,ts in gold_docs())):
        if d not in raw or raw[d]["fold"]!="new":continue
        r=raw[d];full,di=restore(r,tr,el,ts);unseen,_=restore(r,tr,el,ts,seen,minimum_readers=1)
        # Keep the same four-reader-qualified mention universe for the new-reader sensitivity.
        all_obs[d]=full;fresh[d]={q:v for q,v in unseen.items() if q in full};counts.update(di)
    if len(all_obs)!=201:raise ValueError("New panels incomplete")
    return all_obs,fresh,counts

def article(r,annotations,raw):
    d=r["id"];works=defaultdict(list);count=Counter();words=[r["text"][a:b].casefold() for a,b in r["words"]]
    ix=raw["q"];acts=raw["act"];z=raw["words"];cut=0
    if not set(map(int,annotations)).issubset(ix):raise ValueError("Unforecast noun in grading")
    for ii,q in enumerate(ix):
        p=z[:,cut:cut+q].astype(float);cut+=int(q)
        if str(q) not in annotations:continue
        a=annotations[str(q)];y=np.array(a["action"]);w=y/y.sum()
        actions=-np.log(acts[ii])@w;joint=actions.copy();always=np.zeros(len(ARMS))
        count["decisions"]+=int(y.sum());count["clicks"]+=int(y[0]);count["mentions"]+=1
        if a["clicks"]:
            vs=[];hs=[];weights=[]
            for lo,hi,n in a["clicks"]:
                vs.append(-np.logaddexp.reduce(p[:,lo:hi],axis=1));mask=(np.arange(q)>=lo)&(np.arange(q)<hi)
                hs.append([hit_k(v,mask,1) for v in p]);weights.append(n)
            likelihood=np.average(vs,weights=weights,axis=0);top=np.average(hs,weights=weights,axis=0)
            joint+=w[0]*likelihood;always=w[0]*top
            for key in ("all_links","earlier_copy_exists" if words[q] in words[:q] else "no_earlier_copy"):
                works[key+"/range_nll"].append(likelihood);works[key+"/top_word"].append(top)
        for k,v in (("action_nll",actions),("full_action_and_link_nll",joint),("unconditional_word_hit",always)):works[k].append(v)
    return works,count

def graded(rows,data,read_article):
    groups=defaultdict(lambda:defaultdict(list));counts=Counter();docs=0;ww=set()
    for r in rows:
        d=r["id"]
        values,n=read_article(r,data[d])
        if n["mentions"]:docs+=1;ww.add(r["group"])
        counts.update(n)
        for name,items in values.items():groups[name][r["group"]].extend(items)
    rng=np.random.default_rng(49182);scores={}
    for ep,g in sorted(groups.items()):
        x=np.stack([np.mean(v,axis=0) for _,v in sorted(g.items())]);n=len(x);w=rng.multinomial(n,np.ones(n)/n,size=4000)/n;b=w@x
        def ci(v,boot):return [float(v),*np.quantile(boot,[.025,.975]).tolist()]
        res={k:ci(x[:,j].mean(),b[:,j]) for j,k in enumerate(ARMS)}
        for i,j in ((3,0),(3,1),(3,2),(1,0)):res[ARMS[i]+"_minus_"+ARMS[j]]=ci((x[:,i]-x[:,j]).mean(),b[:,i]-b[:,j])
        scores[ep]={"independent_works":n,"estimators":res}
    return {"articles":docs,"works":len(ww),"counts":dict(counts),"scores":scores}

def grade():
    manifest=json.loads((OUT/"copy-external-forecast.json").read_text())
    if manifest["gate"]!=digest(OUT/"copy-fit.json"):raise ValueError("Source gate changed")
    new=[r for r in originals() if r["fold"]=="new"];all_obs,fresh,counts=reveal()
    overlap=json.loads(Path("research/results/reading-validation-transition-pointer/joint-actions/independent-article-actions.json").read_text())["material_overlap"]
    clean=[r for r in new if not any(overlap[r["id"]][k] for k in ("shares_work_known_gold","shares_20_words_known_gold"))]
    def read_article(row, annotations):
        path=CACHE/(row["id"]+".npz")
        if digest(path)!=manifest["files"][path.name]:raise ValueError("Frozen article forecasts changed")
        with np.load(path) as pred:return article(row,annotations,pred)
    scores={"original_intake":graded(new,all_obs,read_article),"unseen_work_and_no_shared_twenty_words":graded(clean,all_obs,read_article),
            "unseen_readers_on_those_new_works":graded(clean,fresh,read_article)}
    write_json(OUT/"new-work-scores.json",{"forecast":digest(OUT/"copy-external-forecast.json"),"eligible_article_pool":len(new),
                "previously_unseen_article_pool":len(clean),"source_audit":dict(counts),"groups":scores})

def old_gold():
    from research.reading_validation.referent_sketch_prepare import material
    intake=json.loads((OUT/"observations.json").read_text())
    fit=json.loads((OUT/"fit.json").read_text())
    source=ROOT/"nominal-external-observations.json"
    if digest(source)!=intake["files"]["external"] or any(digest(OUT/name)!=code for name,code in fit["files"].items()):
        raise ValueError("The original noun data or forecasts changed")
    rows=[row for row in material()["pointer"] if row["fold"]=="test"]
    loc=json.loads((OUT/"queried-positions.json").read_text())
    _,_,_,gate=model_weights()
    with np.load(OUT/"predictions.npz") as archive:
        def read_article(row,annotations):
            uid=row["id"]
            qs=loc[uid]
            if not qs:
                if annotations:raise ValueError("Held noun missing its forecast")
                return {},Counter()
            forms=np.array([row["text"][a:b].casefold() for a,b in row["words"]])
            actions=archive[uid+".a"][:,[0,1,1,1],:]
            original=archive[uid+".r"].astype(float)
            pos=0
            logp=[]
            for q in qs:
                logp.append(log_forecasts(forms,q,*original[:,pos:pos+q],gate))
                pos+=q
            if pos!=original.shape[1]:raise ValueError("Gold candidate geometry changed")
            return article(row,annotations,dict(q=qs,act=actions,words=np.concatenate(logp,axis=1)))
        result=graded(rows,json.loads(source.read_text()),read_article)
    write_json(OUT/"copy-gold-followup.json",dict(previously_examined=True,source_fit=digest(OUT/"copy-fit.json"),results=result))

if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("mode",choices=("forecast","score","old-gold"));cli.add_argument("--native",type=Path);cli.add_argument("--previous",type=Path)
    a=cli.parse_args()
    if a.mode=="forecast":forecast(a.native,a.previous)
    elif a.mode=="score":grade()
    else:old_gold()
