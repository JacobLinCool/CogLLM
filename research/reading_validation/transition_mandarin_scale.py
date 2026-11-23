"""Diagnose base-rate shift in transferred Mandarin choices without adapting any lexical slope."""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from research.reading_validation.transition_mandarin_prior_prepare import DEST, ROOT
from research.reading_validation.transition_score import dump,sha

PROTOCOL=Path("research/reading_validation/transition-mandarin-scale-protocol.md")


def calibrate(z,other,weights):
    n=len(z);W=weights[:,None,:]*(1-np.eye(n))[None,:,:];W/=W.sum(2,keepdims=True)
    intercept=logit(np.einsum("bij,j->bi",W,other))-np.einsum("bij,j->bi",W,z)
    for _ in range(12):
        p=expit(intercept[:,:,None]+z[None,None,:]);g=(W*(p-other)).sum(2);curvature=(W*p*(1-p)).sum(2)
        delta=g/curvature;intercept-=delta
        if abs(delta).max()<1e-10:break
    if abs((W*(expit(intercept[:,:,None]+z[None,None,:])-other)).sum(2)).max()>1e-9:
        raise ArithmeticError("A response intercept failed to converge")
    return expit(intercept+z[None,:])


def main():
    norm=json.loads((ROOT/"mandarin-original-norm-outcomes.json").read_text());src=json.loads((DEST/"seal.json").read_text())
    results=json.loads((DEST/"scores.json").read_text());file=DEST/"frozen-lexical-forecasts.npz"
    if sha(file)!=results["lexical_forecasts_sha256"]:raise ValueError("Source lexical estimates changed")
    vocabulary=[json.loads(s)["verb"] for s in (ROOT/"mandarin-lexical-prior-stimuli.jsonl").read_text().splitlines()]
    names=sorted(norm["xia"]);test=[vocabulary.index(v) for v in names];tr=[vocabulary.index(v) for v in src["source_fitted_verbs"]]
    with np.load(file) as preds:
        logs={k:logit(preds[k][test]) for k in preds.files};means={k:float(logit(preds[k][tr]).mean()) for k in preds.files}
    y={"svo":np.array([norm["xia"][k]["svo_subject"] for k in names]),"scov":np.array([norm["xia"][k]["scov_subject"] for k in names])}
    result={}
    for target,other in (("svo","scov"),("scov","svo")):
        obs=y[target];cal=y[other];freq=logs["training_mean"]
        def score(w):
            q={"neutral":np.full((len(w),len(obs)),.5),"source_mean":np.broadcast_to(expit(freq),w.shape),
               "calibrated_local_mean":calibrate(np.zeros_like(freq),cal,w)}
            for name in ("input_embeddings","orthography","isolated_states"):
                q[name+"_frozen"]=np.broadcast_to(expit(logs[name]),w.shape)
                q[name+"_neutral"]=np.broadcast_to(expit(logs[name]-means[name]),w.shape)
                q[name+"_calibrated"]=calibrate(logs[name],cal,w)
            w=w/w.sum(1,keepdims=True)
            out={}
            for method,p in q.items():
                out[method]=np.column_stack(((w*(-obs*np.log(p)-(1-obs)*np.log1p(-p))).sum(1),
                      (w*(obs*(1-p)**2+(1-obs)*p**2)).sum(1),(w*(p-obs)**2).sum(1),(w*(p-obs)).sum(1)))
            return out,q
        unit=np.ones((1,len(obs)));point,q=score(unit);rng=np.random.default_rng(7004);samples={k:[] for k in q}
        for _ in range(80):
            a,_=score(rng.exponential(size=(50,len(obs))))
            for k,v in a.items():samples[k].append(v)
        boot={k:np.concatenate(z) for k,z in samples.items()}
        def measure(p,b):
            return {k:[float(v),*np.quantile(z,[.025,.975]).tolist()] for k,v,z in zip(("logloss","reader_brier","proportion_mse","mean_overprediction"),p,b.T,strict=True)}
        comparisons=[("input_embeddings_neutral","neutral"),("input_embeddings_neutral","orthography_neutral"),
                     ("input_embeddings_calibrated","calibrated_local_mean"),("input_embeddings_calibrated","orthography_calibrated"),
                     ("input_embeddings_frozen","neutral")]
        result[target]={"observed_equal_verb_first_rate":float(obs.mean()),"other_construction_rate":float(cal.mean()),
                        "methods":{k:measure(v[0],boot[k]) for k,v in point.items()},
                        "paired":{a+"_minus_"+c:measure(point[a][0]-point[c][0],boot[a]-boot[c]) for a,c in comparisons},
                        "per_verb":{m:dict(zip(names,v[0].tolist(),strict=True)) for m,v in q.items()}}
    dump(DEST/"between-construction-calibration.json",{"protocol_sha256":sha(PROTOCOL),"original_result_sha256":sha(DEST/"scores.json"),
         "distinct_verbs":len(names),"held_construct_groups":2,"cohorts":result})
    for k,v in result.items():print(k,{m:r["logloss"] for m,r in v["paired"].items()},flush=True)


if __name__=="__main__":main()
