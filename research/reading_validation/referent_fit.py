"""Fit shared and private earlier-word readouts without opening external response tables."""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.referent_estimation import DIM,Clicks,Twins,fit,gradient_audit,is_development
from research.reading_validation.referent_sketch_prepare import material,OUT,POINTER,PROTOCOL,ROOT
from research.reading_validation.transition_pointer_features import CONTROL,FORMS,design

def anchor_indices(r):
    return [i for i,(a,b) in enumerate(r["words"]) if i and r["text"][a:b].casefold() in FORMS]

def projects(rows,raw):
    n=0;total=None;squares=None
    for r in rows:
        h=raw[r["id"]+"__h"].astype(float)
        if total is None:
            total=np.zeros(h.shape[1]);squares=np.zeros((h.shape[1],h.shape[1]))
        n+=len(h);total+=h.sum(0);squares+=h.T@h
    mu=total/n
    eig,vec=np.linalg.eigh(squares/n-np.outer(mu,mu))
    order=np.argsort(-eig)[:DIM]
    if eig[order].min()<=0:raise ValueError("Singular source representation")
    return mu,vec[:,order]/np.sqrt(eig[order]),n

def extract_features(rows,raw,center,basis,surface):
    data={}
    for r in rows:
        h=(raw[r["id"]+"__h"].astype(float)-center)@basis
        bz=raw[r["id"]+"__b"];cursor=0;prior={}
        if r["id"]+"__h" not in raw.files:raise ValueError("Missing source-independent extraction")
        needs=r["queries"] if "measurement" not in r else [r["measurement"]["q"]]
        locations={q:i for i,q in enumerate(r["queries"])}
        for q in anchor_indices(r):
            if q in needs:
                prior[q]=design(r,q,locations[q])@surface+bz[cursor:cursor+q]
            cursor+=q
        if cursor!=len(bz) or set(prior)!=set(needs):raise ValueError("Anchor/candidate coverage failed")
        data[r["id"]]={"h":h,"b":prior}
    return data

def traces(f,q,last,methods):
    h=f["h"]
    return np.stack([f["b"][q]+(h[q]@W.T+(h[q:last+1]-h[q])@H.T)@h[:q].T for W,H in methods.values()])

def run(source):
    receipt=json.loads((source/"receipt.json").read_text())
    seal=json.loads((OUT/"exposure-seal.json").read_text())
    cache=source/"word-states.npz"
    if (seal["input_sha256"]!=receipt["input_sha256"] or receipt["forwards"]!=seal["articles"] or digest(cache)!=receipt["archive_sha256"]
        or receipt["external_outcomes"]!=0 or digest(POINTER)!=seal["source_weights_sha256"] or digest(PROTOCOL)!=seal["protocol_sha256"]):
        raise ValueError("Native source/outcome separation or archive integrity failed")
    studies=material()
    if any(digest(ROOT/(s+"-stimuli.jsonl"))!=seal["material_sha256"][s] for s in studies):raise ValueError("Study packet changed")
    pd,dp=studies["pointer"],studies["dpr-pointer"]
    ctr=[r for r in pd if r["fold"]=="train"];cdev=[r for r in pd if r["fold"]=="dev"]
    dtr=[r for r in dp if r["fold"]=="train" and not is_development(r["group"])];ddev=[r for r in dp if r["fold"]=="train" and is_development(r["group"])]
    with np.load(POINTER) as p:
        slope=(p["headwise__beta"]/p["headwise__sd"])[:len(CONTROL)]
    with np.load(cache) as raw:
        center,basis,n=projects(ctr+dtr,raw)
        allrows=sum(studies.values(),[])
        features=extract_features(allrows,raw,center,basis,slope)
    existing=POINTER.parent/"candidate-word-logits.npz";worst=0.
    with np.load(existing) as e:
        for r in cdev+ [r for r in pd if r["fold"]=="test"]:
            for q,b in features[r["id"]]["b"].items():
                old=e[r["id"]+"/"+str(q)][-1]
                worst=max(worst,float(np.ptp(b-old)))
    if worst>.001:
        raise ValueError(f"Frozen original pointer log odds drifted by {worst}")
    print("Source-only basis and native-click parity verified",worst,flush=True)
    train_clicks=Clicks.make(ctr,features);train_twins=Twins.make(dtr,features)
    checks=gradient_audit(train_clicks,train_twins)
    models,fits=fit(train_clicks,train_twins)
    dev_clicks=Clicks.make(cdev,features);dev_twins=Twins.make(ddev,features)
    dev={name:{"click_nll":float(dev_clicks(w)[0]),"twin_choice_loss":float(dev_twins(w,h)[0])} for name,(w,h) in models.items()}
    weights=OUT/"source-models.npz"
    np.savez_compressed(weights,center=center,basis=basis,**models)
    external={}
    for s,rs in studies.items():
        for r in rs:
            if r.get("fold") in ("train","dev"):continue
            f=features[r["id"]]
            if s=="pointer":
                for q in r["queries"]:external[r["id"]+"/"+str(q)]=traces(f,q,q,models)[:,0]
            else:
                g=r["measurement"];external[r["id"]]=traces(f,g["q"],g["k"],models)
    file=OUT/"forecast.npz"
    np.savez_compressed(file,**external)
    write_json(OUT/"source-fit.json",{"native":receipt,"arms":list(models),"native_parity_max_logodds":worst,"source_basis_words":n,
        "analytic_gradient_check":checks,"optimization":fits,"source_development":dev,"forecast_sha256":digest(file),"weights_sha256":digest(weights),
        "training_click_articles":len(ctr),"training_twins":len(dtr)//2,"development_click_articles":len(cdev),"development_twins":len(ddev)//2})
    print("Shared and private forecasts sealed",dev,flush=True)

if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("source",type=Path);run(p.parse_args().source)
