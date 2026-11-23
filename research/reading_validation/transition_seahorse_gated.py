"""Likelihoods for the original gated English comprehension, repetition and grammar judgments."""

from argparse import ArgumentParser
from collections import Counter
import csv
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from research.reading_validation.memory_data import h
from research.reading_validation.transition_seahorse_data import DEST, EXPECTED, WORK, ZIP, sha
from research.reading_validation.transition_seahorse_features import load, save
from research.reading_validation.transition_seahorse_fit import LABEL, key, linear, solve, weights
from research.reading_validation.transition_seahorse_score import summary

PROTOCOL = Path(__file__).with_name("transition-seahorse-gated-protocol.md")
TASKS = ("comprehension", "repetition", "grammar")
PENALTY = .002


def original(role, *, grades):
    with ZipFile(ZIP) as archive, archive.open(f"seahorse_data/{role}.tsv") as raw:
        for r in csv.DictReader(TextIOWrapper(raw, encoding="utf8", newline=""), delimiter="\t"):
            if r["worker_lang"] != "en-US":
                continue
            columns = ["gem_id", "summary", "model"]
            if grades:
                columns.extend(("question1", "question2", "question3"))
            yield {name:r[name] for name in columns}


def labels(rows):
    y = np.full((len(rows), len(TASKS)), np.nan)
    for i,r in enumerate(rows):
        q1 = r["question1"]
        if q1 in LABEL:
            y[i,0]=LABEL[q1]
        if q1 == "Yes":
            for j in (1,2):
                label=r[f"question{j+1}"]
                if label in LABEL:
                    y[i,j]=LABEL[label]
    return y


def data():
    if sha(ZIP)!=EXPECTED:
        raise ValueError("The original reading archive changed")
    features={}
    seal=load(DEST/"exposure-seal.json")
    for role in seal["packets"]:
        path=WORK/f"seahorse-{role}-document-features.npz"
        if sha(path)!=load(DEST/f"{role}-feature-seal.json")["file"]:
            raise ValueError("Native feature shard failed its original verification")
        with np.load(path,allow_pickle=False) as a:
            for uid, x in zip(a["keys"],a["x"],strict=True):
                if uid in features:
                    raise ValueError("Duplicate cached reading")
                features[uid]=x
    excluded=load(DEST/"word-exposure-audit.json")
    result={}
    for role in ("train","validation","test"):
        held=set(excluded[role]["exclude_overlap_with_held_roles"])
        rows=[r for r in original(role,grades=role!="test") if r["gem_id"] not in held and h(r["summary"]) in features]
        result[role]={"x":np.stack([features[h(r["summary"])] for r in rows]),"rows":rows,
                      "g":np.array([r["gem_id"] for r in rows])}
        if role!="test":
            result[role]["y"]=labels(rows)
    return result


def separate(z,y,g):
    coeff=[]; checks=[]
    for t in range(y.shape[1]):
        mask=np.isfinite(y[:,t])
        coef,check=solve(z[mask],y[mask,t],weights(g[mask]),PENALTY)
        coeff.append(coef);checks.append(check)
    return np.column_stack(coeff),checks


def covariance(z,y,g):
    half=np.array([int(h("seahorse-gated-sharing-v1/"+u)[:8],16)%2 for u in g])
    co=[]; checks=[]
    for side in (0,1):
        subset=half==side
        theta,check=separate(z[subset],y[subset],g[subset])
        co.append(theta[1:]);checks.append(check)
    def normalize(x):
        length=np.linalg.norm(x,axis=0)
        if np.any(length<1e-8):
            raise ValueError("Unidentified source task slopes")
        return x/length
    reproducibility=np.clip((normalize(co[0])*normalize(co[1])).sum(axis=0),0,1)
    mean=normalize(np.mean(co,axis=0))
    shared=mean.T@mean
    root=np.sqrt(reproducibility)
    R=shared*root[:,None]*root[None,:]+np.diag(1-reproducibility)
    R=.999*R+.001*np.eye(len(TASKS))
    if np.min(np.linalg.eigvalsh(R))<=0 or not np.allclose(R.diagonal(),1):
        raise ArithmeticError("Task covariance is not a proper correlation matrix")
    return R,dict(reproducibility=reproducibility.tolist(),mean_pattern_cosines=shared.tolist(),
                  coefficient_correlation=R.tolist(),source_half_convergence=checks)


def joint(z,y,g,R,start):
    w=np.zeros_like(y); clean=np.nan_to_num(y,nan=0.)
    for j in range(w.shape[1]):
        mask=np.isfinite(y[:,j]); w[mask,j]=weights(g[mask])
    precision=np.linalg.inv(R)
    def criterion(flat):
        theta=flat.reshape(start.shape);B=theta[1:]
        eta=z@B+theta[0]
        residual=w*(expit(eta)-clean)
        penalty_gradient=PENALTY*(B@precision)
        cost=float(np.sum(w*(np.logaddexp(0,eta)-clean*eta))+.5*np.sum(B*penalty_gradient))
        grad=np.vstack((np.sum(residual,axis=0),z.T@residual+penalty_gradient))
        return cost,grad.ravel()
    fit=minimize(criterion,start.ravel(),jac=True,method="L-BFGS-B",options={"maxiter":1000,"gtol":1e-8,"ftol":1e-14,"maxls":50})
    residual=float(np.max(abs(fit.jac))); initial=criterion(start.ravel())[0]
    if residual>2e-6 or not np.isfinite(fit.fun) or fit.fun>initial+1e-9:
        raise ArithmeticError(f"Joint gradient {residual}: {fit.message}")
    return fit.x.reshape(start.shape),dict(initial=initial,objective=float(fit.fun),max_gradient=residual,updates=int(fit.nit))


def fit():
    split=data();tr,va,te=(split[k] for k in ("train","validation","test"))
    keys=[key(r) for r in te["rows"]]
    if len(keys)!=len(set(keys)):
        raise ValueError("Ambiguous held ballot identity")
    parameters={};pred={"keys":np.array(keys)};fits={}
    for typ,width in (("form",15),("native",187)):
        x=tr["x"][:,:width]
        # Match the original Q1 normalization, excluding the three indeterminate comprehension responses.
        yesno=np.isfinite(tr["y"][:,0]); w=weights(tr["g"][yesno]);center=w@x[yesno]
        scale=np.sqrt(w@(x[yesno]-center)**2);scale[scale<1e-7]=1.
        z=(x-center)/scale
        zv=(va["x"][:,:width]-center)/scale
        zt=(te["x"][:,:width]-center)/scale
        separate_co,independent=separate(z,tr["y"],tr["g"])
        R,related=covariance(z,tr["y"],tr["g"])
        joint_co,converged=joint(z,tr["y"],tr["g"],R,separate_co)
        fits[typ]={"covariance":related,"independent":independent,"joint":converged,"calibration":{}}
        parameters.update({typ+".center":center,typ+".scale":scale,typ+".R":R})
        for method,co in (("independent",separate_co),("adaptive",joint_co)):
            name=f"{typ}_{method}";val=zv@co[1:]+co[0];test=zt@co[1:]+co[0]
            probabilities=[];calibrators=[];checks=[]
            for j in range(len(TASKS)):
                use=np.isfinite(va["y"][:,j]);b,check=solve(val[use,j,None],va["y"][use,j],weights(va["g"][use]),1e-8,positive=True)
                probabilities.append(expit(linear(test[:,j,None],b)));calibrators.append(b);checks.append(check)
            pred[name]=np.column_stack(probabilities);parameters[name+".theta"]=co;parameters[name+".cal"]=np.stack(calibrators)
            fits[typ]["calibration"][method]=checks
    # Cross-check the original Q1 predictions before any secondary held grades are read.
    previous=DEST/"forecasts-before-held-responses.npz"
    if sha(previous)!=load(DEST/"fit-seal.json")["forecast_sha256"]:
        raise ValueError("Original Q1 forecast changed")
    replay={}
    with np.load(previous,allow_pickle=False) as a:
        if not np.array_equal(a["keys"],pred["keys"]):
            raise ValueError("Original held presentations differ")
        for typ in ("form","native"):
            err=float(np.max(abs(a[typ]-pred[typ+"_independent"][:,0])))
            if err>2e-6:
                raise ArithmeticError(f"Independent Q1 comparison diverges from original observer: {err}")
            replay[typ]=err
    forecast=DEST/"gated-forecasts-before-secondary-responses.npz"
    weights_file=DEST/"gated-weights.npz"
    np.savez_compressed(forecast,**pred);np.savez_compressed(weights_file,**parameters)
    save(DEST/"gated-fit-seal.json",{"source":EXPECTED,"protocol":sha(PROTOCOL),"uninspected_test_questions":["Q2","Q3"],
         "previously_evaluated_test_question":"Q1","fit_rows":{r:int(len(v["g"])) for r,v in split.items()},
         "source_definitive_masks":{r:np.isfinite(split[r]["y"]).sum(axis=0).tolist() for r in ("train","validation")},
         "questions":list(TASKS),"Q1_independent_original_max_difference":replay,"optimizer":fits,
         "forecast_sha256":sha(forecast),"weights_sha256":sha(weights_file)})
    print("Sealed",len(keys),replay,{k:fits[k]["covariance"]["coefficient_correlation"] for k in fits},flush=True)


def answers_report(rows):
    result={}
    for q in range(1,4):
        observed=Counter(r[f"question{q}"] for r in rows)
        after_no=Counter(r[f"question{q}"] for r in rows if r["question1"]=="No")
        after_unsure=Counter(r[f"question{q}"] for r in rows if r["question1"]=="Unsure")
        result[f"Q{q}"]={"original_cells":dict(observed),"after_Q1_No":dict(after_no),"after_Q1_Unsure":dict(after_unsure)}
    return result


def compare_predictions(rows,pred):
    y=labels(rows);mask=np.isfinite(y);clean=np.nan_to_num(y);g=np.array([r["gem_id"] for r in rows])
    articles,ix=np.unique(g,return_inverse=True);G=len(articles)
    counts=np.random.default_rng(456214).multinomial(G,np.full(G,1/G),size=4000)
    counts=np.vstack((np.ones(G),counts))
    def resample(x,selected=None):
        if selected is None:selected=np.ones(len(g),dtype=bool)
        sums=np.bincount(ix[selected],weights=x[selected],minlength=G)
        return (counts@sums)/(counts@np.bincount(ix[selected],minlength=G))
    def pack(z):
        return summary(z[0],z[1:])
    boot={};metrics={};pairs={}
    for name,p in pred.items():
        p=np.clip(p,1e-15,1-1e-15)
        losses=-(clean*np.log(p)+(1-clean)*np.log1p(-p))*mask
        observed=mask.any(axis=1)
        total=resample(losses.sum(axis=1),observed)
        boot[name]={"observed_questions_per_draft":total}
        metrics[name]={"observed_questions_per_draft_nll":pack(total),"tasks":{}}
        for j,t in enumerate(TASKS):
            use=mask[:,j]; logloss=resample(losses[:,j],use);yy=clean[use,j];pp=p[use,j]
            boot[name][t]=logloss
            metrics[name]["tasks"][t]={"observed":int(use.sum()),"no":int(yy.sum()),"mean_predicted_no":float(pp.mean()),
                                      "AUROC":float(roc_auc_score(yy,pp)),"average_precision_no":float(average_precision_score(yy,pp)),
                                      "nll":pack(logloss)}
    for a,b in (("native_adaptive","native_independent"),("form_adaptive","form_independent"),
                ("native_adaptive","form_adaptive"),("native_independent","form_independent")):
        label=a+" minus "+b;pairs[label]={}
        for k in boot[a]:
            diff=boot[a][k]-boot[b][k];d=pack(diff)
            if k in TASKS:
                d["simultaneous_95_three_task_intervals"]=np.quantile(diff[1:],[.025/3,1-.025/3]).tolist()
            pairs[label][k]=d
    return {"scorable_inputs":len(rows),"scorable_with_definitive_question":int(mask.any(axis=1).sum()),
            "original_articles":G,"scored_answer_counts":answers_report(rows),"metrics":metrics,"paired":pairs}


def evaluate():
    path=DEST/"gated-forecasts-before-secondary-responses.npz";seal=load(DEST/"gated-fit-seal.json")
    if sha(path)!=seal["forecast_sha256"] or sha(PROTOCOL)!=seal["protocol"]:
        raise ValueError("Source or projection changed after forecast sealing")
    with np.load(path,allow_pickle=False) as a:
        ids=a["keys"].copy();pred={k:a[k] for k in a.files if k!="keys"}
    full=list(original("test",grades=True));indexed={key(r):r for r in full};rows=[indexed[k] for k in ids]
    result=compare_predictions(rows,pred)
    save(DEST/"gated-results.json",{"source":EXPECTED,"protocol":sha(PROTOCOL),"forecast_lock":sha(path),"official_rows":len(full),
            "original_answer_counts":answers_report(full),**result})
    print("Typed response scores",result["metrics"]["native_adaptive"],result["paired"]["native_adaptive minus native_independent"],flush=True)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__);parser.add_argument("operation",choices=("fit","evaluate"))
    if parser.parse_args().operation=="fit":fit()
    else:evaluate()
