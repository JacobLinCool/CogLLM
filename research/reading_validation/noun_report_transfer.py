"""Weight story comparisons by source reader decisions to revisit an earlier noun."""
import argparse,json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp
from research.backbone_transfer.extract import digest
from research.reading_validation.first_cue import stage,source_schedule
from research.reading_validation.memory_data import BASE,load_rows,sha
from research.reading_validation.memory_fit import counts
from research.reading_validation.memory_score import reveal,summarize
from research.reading_validation.nominal_data import OUT as NOM,ROOT
from research.reading_validation.nominal_estimator import Words
from research.reading_validation.noun_copy import model_weights,query
from research.reading_validation.premise_emission import design,emissions
from research.reading_validation.referent_features import WORD,setup
from research.reading_validation.taxonomic_monitor import Route,normalized,category_features
from research.reading_validation.taxonomic_pairs import Ontology,form,native_words,opportunities,source_links
from research.reading_validation.transition_pointer_features import sentence_ids

OUT=Path("research/results/reading-validation-nominal-reports")
PLAN=Path("research/reading_validation/nominal-report-transfer-protocol.md")

def reader_prior(raw,article,hidden,bag,heads,copy,mu,basis):
    spans=article["words"];row={**article,"sentences":sentence_ids(article["text"],spans)}
    doc=Words(row,(hidden.astype(float)-mu)@basis);start=0;indices=[]
    by_char={tuple(v):i for i,v in enumerate(spans)}
    for sentence in raw:
        indices.append(np.array([by_char[(start+m.start(),start+m.end())] for m in WORD.finditer(sentence)],dtype=int))
        start+=len(sentence)+2
    table={};values={}
    for t,g in enumerate(bag):
        votes=np.empty((len(g.rank),2))
        for j,(prev,a,b) in enumerate(zip(g.origin,g.earlier,g.word,strict=True)):
            qq=int(indices[t][b]);s=int(indices[prev][a])
            if qq not in table:
                action,rank=query(doc,qq,heads,copy)
                table[qq]=np.log(action[[0,3],0])[:,None]+rank[[0,3]]
            votes[j]=table[qq][:,s]
        values[t]=votes
    return values

def covars(route,sentences,bags,weights,arm):
    result=[]
    for i,(words,g) in enumerate(zip(sentences,bags,strict=True)):
        role=route.target(i+1,len(words));binder=np.zeros(8);loc=np.zeros(4)
        if len(g.rank):
            lp=weights[i][:,arm];w=np.exp(lp-logsumexp(lp))
            loc=np.array([w@w,np.bincount(g.origin,w).max(),np.bincount(g.word,w).max(),logsumexp(lp)])
            mu,sd,b,_=route.models["directional"]
            individual=expit(normalized(category_features(g,"directional"),mu,sd)@b)
            relation=np.c_[individual,g.directional[:,0],g.directional[:,1],g.symmetric[:,0]]
            binder=np.r_[w@relation,relation[np.argmax(w)]]
        result.append(np.r_[design(i,words),role,loc,binder,role*binder])
    return np.asarray(result)

def run(native,original):
    import torch
    torch.set_num_threads(1);setup();folder=OUT/"forecast";folder.mkdir(exist_ok=True,parents=True)
    sig=json.loads((NOM/"report-exposure.json").read_text());cap=json.loads((native/"receipt.json").read_text())
    material=ROOT/"nominal-report-texts.jsonl";states=native/"word-states.npz"
    if digest(material)!=sig["source"] or sig["protocol"]!=digest(PLAN) or cap["input_sha256"]!=sig["source"] or cap["archive_sha256"]!=digest(states):raise ValueError("Unsealed nominal memory materials")
    rows=json.loads((BASE/"memory/study-map.json").read_text());raw={r["id"]:r["sentences"] for r in load_rows(BASE/"memory/stories.jsonl")}
    docs={k:list(map(form,v)) for k,v in raw.items()};natives=native_words(original,raw);ont=Ontology()
    bags={k:opportunities(v,natives[k],ont) for k,v in raw.items()}
    mu,basis,heads,copy=model_weights();pr={}
    with np.load(states) as h:
        for d in map(json.loads,material.read_text().splitlines()):
            k=d["id"];pr[k]=reader_prior(raw[k],d,h[k+"__h"],bags[k],heads,copy,mu,basis)
    source=[r for r in rows if r["study"] in (1,2)];e2=[r for r in source if r["study"]==2]
    observations=counts(reveal(2,rows),e2);emobs={(2,k):v for k,v in observations.items()};links=source_links(source,raw)
    @lru_cache(None)
    def router(excl):return Route([r for r in source if r["story"] not in excl],docs,bags,links)
    forecasts={};history={}
    for held in sorted({r["story"] for r in e2}):
        train=[r for r in e2 if r["story"]!=held];fitrows=[r for r in source if r["story"]!=held];outer=[r for r in rows if r["study"]==3 and r["story"]==held]
        em,_=emissions(fitrows,emobs,docs);history[held]={}
        for i,m in enumerate(("surface_readers","native_readers")):
            X={};prior={};hv={}
            for r in train+outer:
                k=r["id"];excl=tuple(sorted({r["story"],held}))
                X[k]=covars(router(excl),docs[k],bags[k],pr[k],i)
                prior[k]=source_schedule([z for z in source if z["story"] not in excl],len(docs[k]));hv[k]=em(docs[k])
            for moment in (False,True):
                arm=m+("_moment" if moment else "")
                pred,d=stage(train,outer,X,prior,observations,hv,moment);history[held][arm]=d
                for k,(z,_) in pred.items():forecasts["strict."+arm+"."+k]=z
        print("Transferred actual noun responses, held family",held,flush=True)
    old=Path("research/results/reading-validation-first-cue/forecast")
    if digest(old/"hazards.npz")!=json.loads((old/"seal.json").read_text())["files"]["hazards.npz"]:raise ValueError("Historical forecasts changed")
    with np.load(old/"hazards.npz") as h:
        for k in h.files:forecasts[k]=h[k]
    np.savez_compressed(folder/"hazards.npz",**forecasts)
    (folder/"fits.json").write_text(json.dumps(history,indent=2)+"\n")
    (folder/"seal.json").write_text(json.dumps({"protocol":digest(PLAN),"native":cap["archive_sha256"],"noun_weights":digest(NOM/"source-models.npz"),
        "gate":digest(NOM/"copy-gate.npz"),"third_cohort_in_training":False,"files":{n:digest(folder/n) for n in ("fits.json","hazards.npz")}},indent=2)+"\n")

def grade():
    folder=OUT/"forecast";seal=json.loads((folder/"seal.json").read_text())
    if any(digest(folder/n)!=v for n,v in seal["files"].items()):raise ValueError("Altered source forecast")
    meta=json.loads((BASE/"memory/study-map.json").read_text());obs=reveal(3,meta)
    arms=("directional","first_cue","first_cue_moment","moment","oracle","surface_readers","native_readers","surface_readers_moment","native_readers_moment")
    with np.load(folder/"hazards.npz") as pred:
        score=summarize(pred,obs,376,cohorts=("strict",),methods=arms,focus="native_readers",controls=tuple(m for m in arms if m!="native_readers"))
    (OUT/"scores.json").write_text(json.dumps(score,indent=2,allow_nan=False)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("fit","score"));p.add_argument("--native",type=Path);p.add_argument("--original",type=Path)
    a=p.parse_args()
    if a.action=="fit":run(a.native,a.original)
    else:grade()
