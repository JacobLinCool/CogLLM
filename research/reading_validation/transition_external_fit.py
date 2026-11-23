"""Train interval-aligned coherence readers, seal document predictions, and expose thresholds."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.reading_validation.transition_external_outcomes import votes
from research.reading_validation.transition_external_prepare import OUT
from research.reading_validation.transition_score import apply, dump, sha

COHE = "cohesentia"
FROZEN = Path("research/results/reading-validation-transition/readouts.json")
COHE_PROTOCOL = "caeddcc8393e0397dcab74d165f1f59101b40e0b6676ab859cf81a2822f05fc5"


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


def binomial(x, positives, total):
    if x.shape[0] != len(total) or not total.min() > 0:
        raise ValueError("Every fitted row needs observed judgments")
    mean = x.mean(axis=0)
    sd = np.maximum(x.std(axis=0), 1e-10)
    z = (x-mean)/sd
    z = np.column_stack((np.ones(len(x)), z))
    penalty = np.r_[0., np.repeat(.01, len(mean))]
    coef = np.zeros(z.shape[1])
    n = total.sum()
    def loss(beta):
        logits = z @ beta
        return (np.logaddexp(0.,logits) @ total - logits @ positives)/n + .5 * np.sum(penalty*beta**2)
    steps = []
    for _ in range(80):
        eta = z @ coef
        p = sigmoid(eta)
        grad = z.T @ (total*p-positives)/n + penalty*coef
        norm = float(abs(grad).max())
        steps.append({"loss":float(loss(coef)), "max_abs_gradient":norm})
        if norm <= 1e-9:
            break
        hessian = z.T @ ((total*p*(1-p))[:,None]*z)/n + np.diag(penalty)
        step = np.linalg.solve(hessian, grad)
        size = 1.
        while loss(coef-size*step) > loss(coef) - 1e-4*size*np.dot(grad,step):
            size*=.5
            if size<1e-8:
                raise ArithmeticError("Line search failed before convergence")
        coef-=size*step
    if steps[-1]["max_abs_gradient"] > 1e-9:
        raise ArithmeticError("Unconverged coherence readout")
    return {"mean":mean.tolist(), "std":sd.tolist(), "weights":coef[1:].tolist(),
            "intercept":float(coef[0]), "convergence":steps}


def predict(x, model):
    z = (x-np.asarray(model["mean"])) / np.asarray(model["std"])
    return sigmoid(z@np.asarray(model["weights"])+model["intercept"])


def read_features(directory):
    meta=json.loads((OUT/"stimuli.json").read_text())
    if meta["protocol_sha256"]!=COHE_PROTOCOL:
        raise ValueError("External protocol changed")
    receipt=json.loads((directory/"receipt.json").read_text())
    feature_path=directory/"features.jsonl"
    if (receipt["input_sha256"]!=meta["packet_sha256"] or receipt["documents"]!=meta["docs"] or
            sha(feature_path)!=receipt["files"][feature_path.name] or
            receipt["raw_forward_calls"]!=meta["docs"] or receipt["directed_forward_calls"]!=meta["docs"]):
        raise ValueError("Feature and single-pass receipt mismatch")
    records = meta["material"]
    rows = [json.loads(line) for line in feature_path.open()]
    values = {x["id"]: x for x in rows}
    if len(values)!=len(rows) or set(values)!={x["id"] for x in records}:
        raise ValueError("Every scored boundary needs one feature row")
    observed=np.stack([values[r["id"]]["features"] for r in records])
    reps=np.stack([values[r["id"]]["representation"] for r in records])
    if reps.shape[1] % 3 or any(values[r["id"]]["position"]!=r["index"] for r in records):
        raise ValueError("Inconsistent representations or positions")
    directed=np.array([values[r["id"]]["directed"] for r in records])
    position=np.log1p(np.array([r["index"] for r in records]))
    features=np.column_stack((position,observed,reps,directed))
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite forecast input")
    source_stop=10
    end=source_stop+reps.shape[1]
    width=reps.shape[1]//3
    arms={"lexical":[0,1,2,3,4], "display":[0,5], "lexical_surprisal":[0,1,2,3,4,6],
          "local_state":[0,1,2,6]+list(range(source_stop,source_stop+width)),
          "integration":list(range(end)), "directed":[end]}
    return records,observed,features,arms


def train_and_lock(directory):
    records,observed,features,arms=read_features(directory)
    model_dict = {}
    pred = pd.DataFrame([{k:r[k] for k in ("id","source","split","group","index")} for r in records])
    train,dev = [(pred.source.eq(COHE)&pred.split.eq(fold)).to_numpy(copy=True) for fold in ("train","dev")]
    votes_by_row = np.zeros((len(records),2),dtype=int)
    skipped = {}
    for i,r in enumerate(records):
        if train[i] or dev[i]:
            pos,tot,_,wrong=votes(r)
            votes_by_row[i] = (pos,tot)
            if wrong:
                skipped[r["group"]] = wrong
    train &= votes_by_row[:,1]>0
    dev &= votes_by_row[:,1]>0
    pos,denom = votes_by_row.T
    thresholds = {}
    for name,cols in arms.items():
        model=binomial(features[train][:,cols],pos[train],denom[train])
        values=predict(features[:,cols],model)
        valid_neg=np.repeat(values[dev],(denom[dev]-pos[dev]).astype(int))
        threshold=float(np.quantile(valid_neg,.9,method="higher"))
        model_dict[name] = model
        thresholds[name] = threshold
        pred[name]=values
    # This student receives teacher probabilities exclusively on human training stories.
    # All evaluation predictions use the same raw, single-pass article features as integration.
    credit=predict(features[train][:,arms["directed"]],model_dict["directed"])
    features_raw=features[:,arms["integration"]]
    student=binomial(features_raw[train],pos[train]+credit,denom[train]+1)
    student_probability=predict(features_raw,student)
    thresholds["student"]=float(np.quantile(np.repeat(student_probability[dev],denom[dev]-pos[dev]),.9,method="higher"))
    model_dict["student"]=student
    pred["student"]=student_probability
    arms["student"]=arms["integration"]
    existing=json.loads(FROZEN.read_text())
    for name in ("lexical","display","integration","lexical_surprisal"):
        cols=existing["arms"][name]
        rating=apply(observed[:,cols],existing["coefficients"]["coherence"][name],"coherence")
        # Deficit below the rating scale ceiling. Threshold comparison remains rating <3.
        pred["carter_"+name]=5-rating
        thresholds["carter_"+name]=2.
    forecast_path=OUT/"heldout-predictions.csv"
    pred.loc[~pred.split.isin(["train","dev"])].to_csv(forecast_path,index=False,float_format="%.15g")
    model_path=OUT/"readouts.json"
    dump(model_path,{"columns":arms,"thresholds":thresholds,"models":model_dict,"source":"CoheSentia official training families",
        "train_groups":int(pred.loc[train,"group"].nunique()),"dev_groups":int(pred.loc[dev,"group"].nunique()),
        "train_observations":int(denom[train].sum()),"dev_observations":int(denom[dev].sum()),
        "ambiguous_annotator_records_by_training_story":skipped,
        "threshold_rule":"score strictly greater than cutoff; new readout 90th percentile (higher) on calibration nonproblem reports"})
    lock={"protocol_sha256":COHE_PROTOCOL,"directed_protocol_sha256":sha(Path("research/reading_validation/transition-directed-protocol.md")),
          "student_protocol_sha256":sha(Path("research/reading_validation/transition-distillation-protocol.md")),
          "external_source_protocol_sha256":sha(FROZEN),"model_sha256":sha(model_path),
          "features_receipt_sha256":sha(directory/"receipt.json"),"forecast_sha256":sha(forecast_path)}
    dump(OUT/"prediction-lock.json",lock)
    return lock


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--features",required=True,type=Path)
    print(train_and_lock(parser.parse_args().features))
