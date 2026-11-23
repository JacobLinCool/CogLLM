"""Cross-fit the five original uncertainty and referent responses across complete constructions and panels."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.reading_validation.transition_ambicoref_prepare import CATEGORIES,OUT,ROOT
from research.reading_validation.transition_ambicoref_score import rank_order,ranking
from research.reading_validation.transition_anchored_scores import relation_scores,traces
from research.reading_validation.transition_score import dump,sha


PROTOCOL=Path("research/reading_validation/transition-ordinal-response-protocol.md")
ARMS=("training_frequency","belief_only","surface_steps","native_update")
METRICS=("five_category_logloss","five_category_brier","ambiguity_auroc","ambiguity_ap","directional_logloss","mean_ambiguity_overprediction")


def family(g,declared):
    if declared!="unmatched":return declared.split("_")[0].split("-")[0]
    # The two unattested abbreviation variants in the full public corpus still expose an overt syntactic construction.
    words=[v.casefold() for v in g["visible_words"]];q=g["q"]
    if words[q-1]=="because":return "IC"
    if any(p in words[:q] for p in ("before","after","when","as","while")):return "TOP"
    return None


def design(g,prior,evidence):
    z=traces(prior,evidence);opts=g["names"];q=g["q"];k=g["k"]
    logs={m:relation_scores(v,opts) for m,v in z.items()}
    belief=logs["native_semantic_update"][-1]
    simple=[belief,abs(belief)]
    dims=[np.log1p(q),np.log1p(k-q),np.log1p(k),q==k,len(opts[0]),len(opts[1]),np.log1p(q-opts[0][-1]),np.log1p(q-opts[1][-1])]
    for m in ("lexical_prior","source_surface","source_steps","lexical_steps_update"):
        x=logs[m];dims.extend([x[0],x[-1],abs(x[-1]),x[-1]-x[0]])
    native=dims.copy()
    for m in ("native_prior","native_semantic_update","native_attention_update","source_semantic","source_attention"):
        x=logs[m];delta=x-x[0];native.extend([x[0],x[-1],abs(x[0]),abs(x[-1]),delta[-1],
                              abs(delta[-1]),abs(delta).max(),np.max(abs(x-x[-1])),np.mean(abs(delta))])
    native.append(abs(logs["native_attention_update"][-1]-logs["native_semantic_update"][-1]))
    return [simple,dims,native]


def fit_logit(Z,yes,n):
    y=np.asarray(yes,float);w=np.asarray(n,float);mask=w>0
    if not np.any(mask) or min(y.sum(),np.sum(w-y))<=0:raise ValueError("Conditional source outcome unsupported")
    x=Z[mask];w=w[mask];y=y[mask]
    def objective(theta):
        v=x@theta[1:]+theta[0];r=w*expit(v)-y
        return (float(w@np.logaddexp(0,v)-y@v+.005*(theta[1:]@theta[1:])),
                np.r_[r.sum(),x.T@r+.01*theta[1:]])
    initial=np.zeros(x.shape[1]+1);sol=minimize(objective,initial,jac=True,method="L-BFGS-B",options={"ftol":1e-15,"gtol":1e-9,"maxiter":650})
    grad=float(np.abs(sol.jac).max())
    if not np.isfinite(sol.fun) or grad>1e-6:raise RuntimeError(f"Five-way observation law failed stationarity, {grad}")
    return sol.x,grad


def predict(train_all,flags,counts):
    raw=counts[flags];row=raw/raw.sum(1,keepdims=True)/len(raw)
    mu=train_all[flags].mean(0);sd=train_all[flags].std(0);sd[sd<1e-8]=1.
    X=(train_all-mu)/sd;train=X[flags]
    n=row.sum(1);amb=row[:,2];direction=row[:,3:].sum(1);left=row[:,:2].sum(1)
    ambiguous,g1=fit_logit(train,amb,n);which,g2=fit_logit(train,direction,n-amb)
    def oriented(x,o):return np.column_stack((x,np.full(len(x),o),o*x))
    conf,g3=fit_logit(np.vstack([oriented(train,-1),oriented(train,1)]),np.r_[row[:,0],row[:,4]],np.r_[left,direction])
    evaluate=lambda x,v:expit(v[0]+x@v[1:])
    p=evaluate(X,ambiguous);side=evaluate(X,which);cl=evaluate(oriented(X,-1),conf);cr=evaluate(oriented(X,1),conf)
    left=(1-p)*(1-side);right=(1-p)*side
    probs=np.column_stack((left*cl,left*(1-cl),p,right*(1-cr),right*cr))
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(1),1) or np.any(probs<=0):
        raise ValueError("Five inferred action probabilities are not a complete nondegenerate distribution")
    return probs,max(g1,g2,g3)


def metrics(p,y,w):
    n=w.sum(1);rows=np.arange(len(y));amb=y==2;direction=~amb
    if not direction.any() or not amb.any():raise ValueError("Held subset lacks a measured category")
    gold=np.eye(5)[y];which=np.where(y>2,p[:,3:].sum(1),p[:,:2].sum(1))
    ll=-np.log(p[rows,y]);b=np.square(gold-p).sum(1)
    au,ap=ranking(*rank_order(p[:,2]),w*amb,w*direction)
    return np.column_stack((w@ll/n,w@b/n,au,ap,(w[:,direction]@(-np.log(which[direction]/(1-p[direction,2]))))/w[:,direction].sum(1),
                w@(p[:,2]-amb)/n))


def summarize(prediction,inc,items,sheets,answer,source,lexical_families,equal_constructions=False):
    # Sort events by original sentence so sheet resampling can preserve sentence-level weights.
    idx=np.flatnonzero(inc);idx=idx[np.argsort(items[idx],kind="stable")];it,sf,y=items[idx],sheets[idx],answer[idx]
    material,mi=np.unique(it,return_inverse=True);sheet_ids,si=np.unique(sf,return_inverse=True)
    group,gi=np.unique(lexical_families[material],return_inverse=True)
    starts=np.r_[0,np.flatnonzero(np.diff(mi))+1]
    factors=np.ones(len(material))
    if equal_constructions:
        norm=Counter(source[material]);factors=np.array([1/norm[v] for v in source[material]])
    def get_weight(sw,lw):
        weights=sw[:,si];sums=np.add.reduceat(weights,starts,axis=1)
        return weights/sums[:,mi]*lw[:,gi[mi]]*factors[mi]
    wp=get_weight(np.ones((1,len(sheet_ids))),np.ones((1,len(group))))
    point={m:metrics(v[idx],y,wp)[0] for m,v in prediction.items()}
    boots={m:[] for m in point};rng=np.random.default_rng(817030);N=3000
    for start in range(0,N,64):
        size=min(64,N-start);w=get_weight(rng.exponential(size=(size,len(sheet_ids))),rng.exponential(size=(size,len(group))))
        for m,v in prediction.items():boots[m].append(metrics(v[idx],y,w))
    mats={m:np.concatenate(z) for m,z in boots.items()}
    for m in ARMS[:-1]:
        key="native_update_minus_"+m;point[key]=point["native_update"]-point[m];mats[key]=mats["native_update"]-mats[m]
    scores={m:{k:[float(x[j]),*np.quantile(mats[m][:,j],[.025,.975]).tolist()] for j,k in enumerate(METRICS)} for m,x in point.items()}
    return {"sentences":len(material),"sheets":len(sheet_ids),"decisions":len(y),"lexical_skeletons":len(group),
            "five_observed_categories":dict(zip(CATEGORIES,np.bincount(y,minlength=5).tolist(),strict=True)),"scores":scores}


def main(folder):
    seal=json.loads((OUT/"external-seal.json").read_text());receipt=json.loads((folder/"receipt.json").read_text())
    gold=ROOT/"ambicoref-observed-choices.json";pfile=folder/"prior-words.npz";dfile=folder/"evidence-words.npz"
    if sha(gold)!=seal["outcomes_sha256"] or sha(pfile)!=receipt["prior_forecast_sha256"] or sha(dfile)!=receipt["evidence_forecast_sha256"]:
        raise ValueError("The original external questionnaire or native forecasts changed")
    gold=json.loads(gold.read_text());d=json.loads((OUT/"design-conditions.json").read_text())["assignments"]
    keys=[k for k,g in gold.items() if family(g,d[k])];families=np.array([family(gold[k],d[k]) for k in keys])
    human=np.array([(i,sheet,c) for i,k in enumerate(keys) for sheet,c in gold[k]["questionnaire_choices"]])
    items,sheet,answer=human.T;sk=np.array([gold[k]["family"] for k in keys])
    if set(families)!={"ECO","ECS","IC","TOP"}:raise ValueError("The complete linguistic families differ")
    features={m:[] for m in ARMS[1:]}
    with np.load(pfile) as p,np.load(dfile) as u:
        for k in keys:
            for m,z in zip(ARMS[1:],design(gold[k],p[k],u[k]),strict=True):features[m].append(z)
    features={m:np.stack(z) for m,z in features.items()};stats=[];forecasts={m:np.full((len(items),5),np.nan) for m in ARMS}
    for fam in sorted(set(families)):
        for fold in range(4):
            tr=(families[items]!=fam)&(sheet%4!=fold);test=(families[items]==fam)&(sheet%4==fold)
            if set(items[tr])&set(items[test]) or set(sheet[tr])&set(sheet[test]):raise ValueError("Training materials or questionnaires leaked")
            counts=np.zeros((len(keys),5),int);np.add.at(counts,(items[tr],answer[tr]),1)
            eligible=counts.sum(1)>0
            frequency=np.mean(counts[eligible]/counts[eligible].sum(1,keepdims=True),axis=0)
            forecasts["training_frequency"][test]=frequency
            fit={}
            for m,X in features.items():
                posterior,grad=predict(X,eligible,counts);forecasts[m][test]=posterior[items[test]];fit[m]=grad
            stats.append({"family":fam,"sheet_fold":fold,"source_sentences":int(eligible.sum()),"source_decisions":int(tr.sum()),
                          "external_sentences":int(len(set(items[test]))),"external_sheets":int(len(set(sheet[test]))),
                          "external_decisions":int(test.sum()),"source_max_abs_gradient":fit})
    if any(not np.isfinite(v).all() for v in forecasts.values()):raise ValueError("An external person response remained unscored")
    file=OUT/"cross_construct_five_way_forecasts.npz";np.savez_compressed(file,items=items,sheets=sheet,categories=answer,**forecasts)
    scores={g:summarize(forecasts,families[items]==g,items,sheet,answer,families,sk) for g in sorted(set(families))}
    full=np.ones(len(items),bool);scores["pooled_sentence_equal"]=summarize(forecasts,full,items,sheet,answer,families,sk)
    scores["four_constructions_equal"]=summarize(forecasts,full,items,sheet,answer,families,sk,True)
    dump(OUT/"cross-construct-five-way.json",{"protocol_sha256":sha(PROTOCOL),"primary_zero_shot_seal_sha256":sha(OUT/"external-seal.json"),
            "forecast_sha256":sha(file),"input_sha256":receipt["input_sha256"],"excluded_unidentified_construction":len(gold)-len(keys),
            "cell_fits":stats,"prediction_features":{m:X.shape[1] for m,X in features.items()},"cohorts":scores})
    print({m:v["scores"]["native_update"]["five_category_logloss"] for m,v in scores.items()},flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--features",type=Path,required=True);main(p.parse_args().features)
