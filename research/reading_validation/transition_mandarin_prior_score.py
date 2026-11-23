"""Predict other Mandarin cohorts' referent choices from nonoverlapping published verb norms."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from research.reading_validation.transition_mandarin_fit import ROOT,OUT,characters,cosine,generate,representation,summarise
from research.reading_validation.transition_mandarin_prior_prepare import DEST,PROTOCOL
from research.reading_validation.transition_score import dump,sha


def logistic(K,idx,y):
    x=representation(K,idx);train=x[idx]
    def objective(w):
        fit=train@w;s=expit(fit);r=w.copy();r[0]=0
        return float(np.mean(np.logaddexp(0,fit)-y*fit)+.005*r@w),train.T@(s-y)/len(y)+.01*r
    opt=minimize(objective,np.zeros(x.shape[1]),method="L-BFGS-B",jac=True,
                 options={"maxiter":600,"ftol":1e-15,"gtol":2e-9,"maxls":50})
    err=float(np.max(abs(opt.jac)))
    if not np.isfinite(opt.fun) or err>1e-6:raise ValueError("Source lexical fit lacks stationarity")
    return expit(x@opt.x),{"iterations":int(opt.nit),"gradient":err,"parameters":len(opt.x),"objective":float(opt.fun)}


def xia_evaluate(vocab,forecasts,observations):
    rng=np.random.default_rng(43435);words=sorted(observations);ii=[vocab.index(v) for v in words]
    draws=rng.integers(len(words),size=(10000,len(words)))
    scores={};paired={}
    for field in ("svo_subject","scov_subject"):
        y=np.array([observations[v][field] for v in words]);et={};pts={};bs={}
        for m,p0 in forecasts.items():
            p=p0[ii];sign=(np.sign(p-.5)+1)/2
            features=np.stack((-y*np.log(p)-(1-y)*np.log1p(-p),y*(1-p)**2+(1-y)*p**2,(p-y)**2,
                               sign*(y>.5)+(1-sign)*(y<.5)+.5*(y==.5)),axis=1)
            names=("choice_logloss","reader_brier","reported_proportion_mse","majority_agreement");point=features.mean(0);boot=features[draws].mean(1)
            def spearman(a,b):
                ar=rankdata(a,axis=-1);br=rankdata(b,axis=-1)
                ac=ar-ar.mean(axis=-1,keepdims=True);bc=br-br.mean(axis=-1,keepdims=True)
                den=np.sqrt(np.sum(ac*ac,axis=-1)*np.sum(bc*bc,axis=-1))
                return np.divide((ac*bc).sum(-1),den,out=np.zeros_like(den,dtype=float),where=den>0)
            eta=float(spearman(p,y));resample=spearman(p[draws],y[draws])
            et[m]={k:[float(t),*np.quantile(b,[.025,.975]).tolist()] for k,t,b in zip(names,point,boot.T,strict=True)}
            et[m]["spearman"]=[eta,*np.quantile(resample,[.025,.975]).tolist()] if len(np.unique(p))>1 else None
            pts[m]=point;bs[m]=boot
        for m in ("input_embeddings","isolated_states"):
            for comp in ("orthography","training_mean"):
                a=pts[m]-pts[comp];b=bs[m]-bs[comp];paired[field+"__"+m+"_minus_"+comp]={k:[float(t),*np.quantile(z,[.025,.975]).tolist()] for k,t,z in zip(names,a,b.T,strict=True)}
        scores[field]=et
    return {"original_participants":166,"verbs":len(words),"decisions_available_as":"rounded proportion by original verb and construction",
            "models":scores,"paired":paired}


def main(lexical,clauses):
    seal=json.loads((DEST/"seal.json").read_text());src=ROOT/"mandarin-original-norm-outcomes.json"
    packet=ROOT/"mandarin-lexical-prior-stimuli.jsonl";lf=lexical/"word-representations.npz";lr=json.loads((lexical/"receipt.json").read_text())
    if (seal["protocol_sha256"]!=sha(PROTOCOL) or sha(src)!=seal["outcomes_sha256"] or sha(packet)!=seal["input_sha256"] or
            lr["input_sha256"]!=sha(packet) or sha(lf)!=lr["weights_sha256"] or lr["dictionary_forwards"]!=188 or lr["external_outcome_fields"]):
        raise ValueError("Independent lexical study provenance changed")
    raw=json.loads(src.read_text());vocab=[json.loads(s)["verb"] for s in packet.read_text().splitlines()]
    ids=[vocab.index(v) for v in seal["source_fitted_verbs"]];y=np.array([np.mean(raw["zhang"][vocab[i]]) for i in ids])
    if set(seal["source_fitted_verbs"])&(set(raw["xia"])|set(raw["zhan_romanzi"].values())):raise ValueError("External verb admitted to source training")
    _,form=characters(vocab)
    with np.load(lf) as f:rawvec=f["features"].astype(float)
    kernels={"orthography":[form],"input_embeddings":[form,cosine(rawvec[:,0])],
             "isolated_states":[form,form,cosine(rawvec[:,1]),cosine(rawvec[:,2])]}
    predictions={"training_mean":np.full(len(vocab),y.mean())};fits={}
    for m,k in kernels.items():predictions[m],fits[m]=logistic(k,ids,y)
    saved=DEST/"frozen-lexical-forecasts.npz";np.savez_compressed(saved,**predictions)
    xia=xia_evaluate(vocab,predictions,raw["xia"])
    prior={a:{code:p[vocab.index(spelling)] for code,spelling in raw["zhan_romanzi"].items()}
           for a,p in predictions.items() if a!="training_mean"}
    f=clauses/"unprompted-states.npz";rec=json.loads((clauses/"receipt.json").read_text())
    if sha(f)!=rec["features_sha256"]:raise ValueError("Original causal article states changed")
    readings=list(map(json.loads,(ROOT/"mandarin-reading-stimuli.jsonl").read_text().splitlines()))
    choices=json.loads((ROOT/"mandarin-anonymous-choices.json").read_text())
    with np.load(f) as cache:cells,info,probs,joint,health,_=generate(readings,choices,cache,lexical_priors=prior)
    comps={a+"_coupled":list(dict.fromkeys(("native_coupled","rated_semantic_class_overt_only",*(("orthography_coupled",) if a!="orthography" else ())))) for a in prior}
    score=summarise(choices,info,probs,joint,additional_comparisons=comps)
    original=json.loads((OUT/"scores.json").read_text())
    np.testing.assert_allclose(score["metrics"]["native_coupled"]["choice_logloss"],original["metrics"]["native_coupled"]["choice_logloss"],atol=1e-12)
    dump(DEST/"scores.json",{"protocol_sha256":sha(PROTOCOL),"lexical_seal_sha256":sha(DEST/"seal.json"),
         "lexical_forecasts_sha256":sha(saved),"norm_training_size":len(ids),"source_fits":fits,
         "dictionary_receipt":lr,"xia":xia,"zhan":score,"zhan_source_largest_gradient":{k:max(h["max_gradient"] for h in v) for k,v in health.items()}})
    print(xia["models"]["svo_subject"],{m:score["metrics"][m+"_coupled"]["choice_logloss"] for m in prior},flush=True)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__);cli.add_argument("--lexical",required=True,type=Path);cli.add_argument("--clauses",required=True,type=Path)
    a=cli.parse_args();main(a.lexical,a.clauses)
