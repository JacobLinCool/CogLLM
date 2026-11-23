"""Learn a contradiction direction and test its relation to first-report hazards."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from research.semantic_tracking import frick_consistency_score as frick
from research.reading_validation.joint_data import DEFAULT_CACHE, digest, lines, read
from research.reading_validation.joint_fit import normalized_hidden


NEGATION = {"not","no","never","nobody","none","nothing","without","neither","nicht","nie","niemand","kein","keine","keinen","ohne","weder"}
WORD = re.compile(r"[^\W_]+",re.UNICODE)


def lexical(premise,hypothesis):
    p,h = [WORD.findall(text.casefold()) for text in (premise,hypothesis)]
    ps, hs = set(p),set(h)
    overlap = len(ps&hs)
    return [np.log1p(len(p)),np.log1p(len(h)),np.log1p(len(hypothesis)),
            np.mean([len(w) for w in h]) if h else 0,
            overlap/max(len(hs),1),overlap/max(len(ps|hs),1),
            float(bool(ps&NEGATION)),float(bool(hs&NEGATION)),
            float(bool(ps&NEGATION)!=bool(hs&NEGATION))]


def fit_binary(x,y):
    mu=x.mean(axis=0)
    scale=x.std(axis=0)
    scale[scale<1e-8]=1.
    design = (x-mu)/scale
    def objective(v):
        logits=design@v[1:]+v[0]
        residual=expit(logits)-y
        return (np.mean(np.logaddexp(0,logits)-y*logits)+.01*np.dot(v[1:],v[1:]),
                np.r_[residual.mean(),design.T@residual/len(y)+.02*v[1:]])
    opt=minimize(objective,np.zeros(design.shape[1]+1),jac=True,method="L-BFGS-B",
                 options=dict(maxiter=1500,gtol=1e-8,ftol=1e-15))
    loss,grad=objective(opt.x)
    if not opt.success or np.max(np.abs(grad))>1e-6:
        raise ArithmeticError(f"Conflict readout has not converged: {opt.message}")
    # Fold training normalization into the exposed linear score.
    slope=opt.x[1:]/scale
    return dict(slope=slope,intercept=float(opt.x[0]-mu@slope),
                gradient_inf=float(np.max(np.abs(grad))),objective=float(loss),iterations=int(opt.nit))


def predict(model,x):
    return x@model["slope"]+model["intercept"]


def classification(y,score):
    y=np.asarray(y,dtype=bool)
    score=np.asarray(score)
    npos=int(y.sum())
    nneg=len(y)-npos
    out=dict(n=len(y),positives=npos)
    if not npos or not nneg:
        return out|dict(ap=None,auc=None)
    ranks=rankdata(score)
    order=np.argsort(-score,kind="stable")
    sorted_y=y[order]
    ends=np.r_[np.diff(score[order])!=0,True]
    totals=np.arange(1,len(score)+1)[ends]
    tp=np.cumsum(sorted_y)[ends]
    ap=(np.diff(np.r_[0,tp])*tp/totals).sum()/npos
    return out|dict(auc=float((ranks[y].sum()-npos*(npos+1)/2)/(npos*nneg)),ap=float(ap))


def fit_xnli(cache, extraction, dest):
    folder=cache/"reading-validation"
    receipt=read(extraction/"extraction.json")
    if receipt["packet_sha256"]!=read(folder/"conflict-input.json")["stimuli_sha256"]:
        raise ValueError("Extraction does not match XNLI inputs")
    for file,expected in receipt["files_sha256"].items():
        if digest(extraction/file)!=expected:
            raise ValueError("XNLI hidden state checksum failed")
    ids=read(extraction/"ids.json")
    ids={key:i for i,key in enumerate(ids)}
    design=read(folder/"conflict-design.json")
    lang=np.array([r["lang"] for r in design]); train=np.array([r["role"]=="dev" for r in design])
    y=np.array([r["label"]=="contradiction" for r in design if r["role"]=="dev"])
    surface=np.asarray([lexical(r["premise"],r["hypothesis"]) for r in design])
    matrix=np.load(extraction/"states.npy",mmap_mode="r")
    hidden=normalized_hidden(matrix[[ids[r["id"]] for r in design]])
    models,diags,weights={},{},{}
    predictions={}
    for method,x in (("surface",surface),("semantic",hidden)):
        m=fit_binary(x[train],y)
        models[method]=m
        weights[method]=m["slope"]
        weights[method+".intercept"]=m["intercept"]
        z=predict(m,x[~train])
        predictions[method]=z
        diags[method]={k:v for k,v in m.items() if k not in ("slope","intercept")}
    test_ids=[r["id"] for r in design if r["role"]=="test"]
    np.savez_compressed(dest/"xnli-heads.npz",**weights)
    np.savez_compressed(dest/"xnli-forecasts.npz",id=test_ids,**predictions)
    seal={k:digest(dest/k) for k in ("xnli-heads.npz","xnli-forecasts.npz")}
    (dest/"xnli-seal.json").write_text(json.dumps(seal,indent=2)+"\n")
    truth=np.array([r["label"]=="contradiction" for r in design if r["role"]=="test"])
    for method,z in predictions.items():
        metrics={}
        for language in ("en","de"):
            keep=lang[~train]==language
            metrics[language]=classification(truth[keep],z[keep])
        diags[method]["test"]=metrics
    return models,diags


def frick_features(cache):
    archive=Path("research/results/frick-consistency-l4")
    seal=read(archive/"extraction.json")
    for f in ("states.npy","coordinates.jsonl"):
        if digest(archive/f)!=seal["file_sha256"][f]:
            raise ValueError("Frick causal feature archive changed")
    states=np.load(archive/"states.npy",mmap_mode="r")
    pos=0
    by_id={}
    for r in lines(archive/"coordinates.jsonl"):
        end=pos+len(r["stops"])
        by_id[r["id"]]=states[pos:end]
        pos=end
    if pos!=len(states):
        raise ValueError("Incomplete Frick coordinate inventory")
    data=cache/"semantic-tracking/frick-consistency/prepared"
    audit,stim,meta,families,texts=frick.load_layout(data)
    old,slots=frick.design_rows(stim,meta,texts,by_id)
    semantic=np.vstack([normalized_hidden(by_id[k][1:]) for k in sorted(meta)])
    return data,audit,meta,families,old[:,:frick.SCALAR_WIDTH],semantic,slots


def summarize_frick(logits,slots,meta,families,trials):
    reader_events=defaultdict(list)
    for key,reader,obs,first in trials:
        reader_events[key].append(first)
    rng=np.random.default_rng(71301)
    methods={}
    family_nll={}
    for method,z in logits.items():
        results=[]
        for name,pair in sorted(families.items()):
            one={"family":name}
            for cond,key in pair.items():
                raw=z[slots[key]]
                log_survival=-np.logaddexp(0,raw)
                before=np.r_[0,np.cumsum(log_survival)[:-1]]
                terminal=np.r_[before-np.logaddexp(0,-raw),log_survival.sum()]
                if abs(np.exp(terminal).sum()-1)>1e-8:
                    raise ArithmeticError("First-event path probabilities do not sum to one")
                t=meta[key]["target_index"]-1
                idx=np.array([len(raw) if i is None else i for i in reader_events[key]])
                one[cond]=dict(n=int(len(idx)),nll=float(-terminal[idx].mean()),
                               human_target=float(np.mean(idx==t)),model_target=float(np.exp(terminal[t])),
                               human_premature=float(np.mean(idx<t)),model_premature=float(np.exp(terminal[:t]).sum()))
            results.append(one)
        family_nll[method]=np.array([(r["01"]["nll"]+r["02"]["nll"])/2 for r in results])
        effects={k:np.array([r["02"][k]-r["01"][k] for r in results]) for k in ("model_target","human_target")}
        methods[method]=dict(
            family_results=results,nll=float(family_nll[method].mean()),
            target_model_delta=float(effects["model_target"].mean()),
            target_human_delta=float(effects["human_target"].mean()),
            target_effect_signed_error=float((effects["model_target"]-effects["human_target"]).mean()),
            premature={c:{k:float(np.mean([r[c][k] for r in results])) for k in ("human_premature","model_premature")}
                       for c in ("01","02")})
    names=list(methods)
    a,b=names[0],names[1]
    sample=rng.integers(0,len(families),size=(10000,len(families)))
    diff=family_nll[b]-family_nll[a]
    def intervals(v):
        return dict(estimate=float(v.mean()),interval95=np.quantile(v[sample].mean(axis=1),[.025,.975]).tolist())
    return dict(methods=methods,comparison=f"{b}_minus_{a}",paired_nll=intervals(diff))


def run(cache,extraction,dest):
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    models,diags=fit_xnli(cache,extraction,dest)
    data,audit,meta,families,surface,features,slots=frick_features(cache)
    trials,_,hits,totals,inventory=frick.load_outcomes(data,audit,meta,slots)
    evidence=predict(models["semantic"],features)
    designs={"position_surface":surface,"plus_semantic_conflict":np.column_stack((surface,evidence))}
    outputs,all_fits={},{}
    for name,x in designs.items():
        z=np.full(len(x),np.nan)
        for group,pair in sorted(families.items()):
            mask=np.zeros(len(x),dtype=bool)
            for key in pair.values():
                mask[slots[key]]=True
            model=frick.fit(x[~mask],hits[1,~mask],totals[1,~mask])
            z[mask]=frick.predict_logits(x[mask],model)
        if not np.isfinite(z).all():
            raise ArithmeticError("Missing out-of-family Frick forecast")
        outputs[name]=z
        fitted=frick.fit(x,hits[1],totals[1])
        all_fits[name]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in fitted.items()}
    np.savez_compressed(dest/"frick-forecasts.npz",**outputs)
    (dest/"frick-calibration.json").write_text(json.dumps(all_fits,indent=2)+"\n")
    result=dict(source=diags,frick_inventory=inventory,frick=summarize_frick(outputs,slots,meta,families,trials),
                protocol_sha256=digest(Path(__file__).with_name("conflict-protocol.md")),
                extraction_sha256=digest(extraction/"extraction.json"),
                validation="retrospective Frick story-family cross-validation; XNLI official test subset")
    (dest/"scores.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"nli":diags,"paired_frick":result["frick"]["paired_nll"]}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    p.add_argument("--extraction",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.cache,a.extraction,a.output)
