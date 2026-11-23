"""Export all previously validated topic-specific observers without choosing an external fold."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_essay_hazard import alpha,basis,learn
from research.reading_validation.transition_essay_reasons import DEST,fit,infer,mass
from research.reading_validation.transition_external_fit import read_features
from research.reading_validation.transition_external_prepare import csv_rows
from research.reading_validation.transition_inference import CHECKPOINT,REVISION
from research.reading_validation.transition_score import dump,sha


def export(folder):
    records,_,features,arms=read_features(folder);rows=[i for i,r in enumerate(records) if r["source"]=="decor"]
    X=features[rows][:,arms["integration"]];data=csv_rows("test_1355_clean.csv")
    raw=[data[records[i]["source_ref"][1]] for i in rows]
    t=np.array([r["topic"] for r in raw]);e=np.array([r["essay_id"] for r in raw]);y=np.array([int(r["label"]=="0.0") for r in raw])
    r=np.array([[0 if not v else int(float(d["R"+str(j)])) for j in range(1,8)] for d,v in zip(raw,y,strict=True)])
    frozen=DEST/"source-anchored-hazard-scores.json";registered=json.loads(frozen.read_text());models=[];err=0.
    pred=np.load(DEST/"source-anchored-hazards.npz")
    for o in registered["folds"]:
        test=t==o["topic"];dev=t==next(f["calibration_topic"] for f in json.loads((DEST/"scores.json").read_text())["folds"] if f["topic"]==o["topic"])
        tr=~(test|dev);w=mass(t[tr],e[tr]);cols=arms["lexical_surprisal"]
        base,_=fit(X[tr][:,cols],y[tr],r[tr],w,False);pb,_=infer(base,X[:,cols]);bm,sd,bw=base
        z,affine=basis(X,np.flatnonzero(tr),w,arms,with_transform=True)
        h=-np.log1p(-pb);cf,pi,_=learn(z[tr],h[tr],r[tr],w)
        g=h*((np.exp(z@cf))@pi);a=alpha(h[dev],g[dev],y[dev],mass(t[dev],e[dev]))
        if abs(a-o["source_reliability"])>1e-9:raise ValueError("Source calibration could not be replayed")
        risk=-np.expm1(-(h[test]+a*(g[test]-h[test])))
        err=max(err,np.max(np.abs(risk-pred["source_reliability"][test])),np.max(abs(pb[test]-pred["lexical_surprisal"][test])))
        base_w=np.zeros(299);base_w[np.asarray(cols)+1]=bw[1:,0]/sd;base_w[0]=bw[0,0]-bm@(bw[1:,0]/sd)
        residual=affine@cf
        np.testing.assert_allclose(np.column_stack((np.ones(len(X)),X))@residual,z@cf,atol=1e-8)
        models.append({"baseline":base_w.tolist(),"residual":residual.tolist(),"reason_prior":pi.tolist(),"reliability":a})
    if err>1e-8:raise ValueError("Shipped expert ensemble differs from the source forecasts")
    path=DEST/"serving.json"
    dump(path,{"schema":"essay-source-anchored-hazard-v1","revision":REVISION,"checkpoint_sha256":CHECKPOINT,
        "source_result_sha256":sha(frozen),"validation_words":"titled English student essays, human coherence codes",
        "validated_sentence_forecasts":len(r),"max_source_replay_error":float(err),"aggregation":"equal mean of eight topic source models",
        "models":models})
    print(sha(path),err,flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--features",type=Path,required=True);export(p.parse_args().features)
