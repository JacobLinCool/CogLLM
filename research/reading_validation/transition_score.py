"""Fit declared Carter–Hoffman readouts on calibration families and score locked forecasts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.reading_validation.transition_outcomes import prepare
from research.reading_validation.transition_prepare import OUT, ROOT

NAMES = ["log_context_words", "log_target_words", "content_overlap", "boundary_overlap", "mean_transition",
         "mean_word_surprisal", "mean_preamble_attention", "mid_cosine", "final_cosine"]
ARMS = {"training_mean": [], "lexical": list(range(4)), "display": [4], "lexical_surprisal": [0,1,2,3,5],
        "without_display": [0,1,2,3,5,6,7,8], "integration": list(range(9))}
ENDPOINTS = {"coherence": "trial_mean_rating", "reading_time": "log_ms"}
PROTOCOL = "68d05c6d016910b318182bda5e7753a0457200538f0b8ec081051534d70564d6"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def setup(features_dir):
    packet = ROOT / "bridge-stimuli.jsonl"
    meta = json.loads((OUT / "stimuli.json").read_text())
    if sha(Path("research/reading_validation/transition-protocol.md")) != PROTOCOL or meta["protocol_sha256"] != PROTOCOL:
        raise ValueError("Protocol digest changed after material preparation")
    if sha(packet) != meta["packet_sha256"]:
        raise ValueError("Material lock changed")
    receipts = json.loads((features_dir / "receipt.json").read_text())
    features = features_dir / "features.jsonl"
    if receipts["features"] != NAMES or receipts["input_sha256"] != sha(packet) or receipts["output_sha256"] != sha(features):
        raise ValueError("Feature provenance mismatch")
    if receipts["forward_calls"] != len(meta["rows"]) or receipts["input_rows"] != len(meta["rows"]):
        raise ValueError("Expected one forward for every passage")
    texts = [json.loads(x) for x in packet.read_text().splitlines()]
    frame, pre, source = prepare(pd.DataFrame(meta["rows"]), texts)
    all_features = [json.loads(x) for x in features.read_text().splitlines()]
    matrices = {row["id"]:np.asarray(row["words"],dtype=float) for row in all_features}
    if len(matrices) != len(all_features) or set(matrices) != set(frame.id):
        raise ValueError("Expected exactly one feature matrix per stimulus")
    for row in frame.itertuples():
        mat = matrices[row.id]
        if mat.shape != (row.target_length,len(NAMES)) or not np.isfinite(mat).all():
            raise ValueError("Invalid matrix")
    return frame.sort_values(["item","condition"]).reset_index(drop=True), pre, matrices, source


def endpoint_rows(frame, endpoint):
    complete = frame.groupby("family")[ENDPOINTS[endpoint]].transform("count").eq(4)
    if not frame.groupby("family").size().eq(4).all():
        raise ValueError("All material groups must contain exactly one quartet for this analysis")
    return frame.loc[complete].copy()


def fit_ridge(x, y):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-12, 1., std)
    z = (x-mean) / std
    intercept = float(y.mean())
    weights = np.linalg.solve(z.T @ z / len(z) + 0.01*np.eye(z.shape[1]), z.T @ (y - intercept) / len(z))
    return {"mean":mean.tolist(),"scale":std.tolist(),"weight":weights.tolist(),"intercept":intercept}


def apply(x, fit, endpoint):
    pred = ((x - np.asarray(fit["mean"])) / np.asarray(fit["scale"])) @ np.asarray(fit["weight"]) + fit["intercept"]
    return np.clip(pred, 1., 5.) if endpoint == "coherence" else pred


def fit_and_lock(frame, matrices, source, feature_dir):
    coefficients = {}
    parts = []
    for endpoint, gold in ENDPOINTS.items():
        data = endpoint_rows(frame, endpoint)
        training = data.split.eq("calibration").to_numpy()
        if len(set(data[training].family) & set(data[~training].family)) != 0:
            raise ValueError("Family leakage")
        if endpoint == "reading_time":
            row_features = np.stack([matrices[r.id][r.critical_index-1] for r in data.itertuples()])
        else:
            row_features = np.stack([matrices[r.id][-1] for r in data.itertuples()])
        holdout = data.loc[~training,["id","item","family","condition"]].copy()
        holdout["endpoint"] = endpoint
        coefficients[endpoint] = {}
        for name, cols in ARMS.items():
            x = row_features[:,cols]
            fit = fit_ridge(x[training], data.loc[training,gold].to_numpy())
            coefficients[endpoint][name] = fit
            holdout[name] = apply(x[~training], fit, endpoint)
        holdout["raw_transition"] = row_features[~training,4]
        parts.append(holdout)
    ck = OUT / "readouts.json"
    dump(ck, {"penalty":0.01, "features":NAMES, "arms":ARMS, "coefficients":coefficients})
    forecasts = OUT / "heldout-predictions.csv"
    pd.concat(parts).to_csv(forecasts, index=False, float_format="%.15g")
    lock = {"protocol_sha256":PROTOCOL, "analysis_details_sha256":sha(Path("research/reading_validation/transition-analysis-notes.md")),
            "source_sha256":sha(Path("research/reading_validation/transition-source.json")),
            "coefficient_sha256":sha(ck), "forecast_sha256":sha(forecasts),
            "source_observations":source, "feature_receipt_sha256":sha(feature_dir / "receipt.json")}
    dump(OUT / "prediction-lock.json", lock)
    return lock


def with_interval(value, boot):
    v = np.asarray(boot)
    valid = np.isfinite(v)
    return {"estimate":float(value) if np.isfinite(value) else None,
            "family_bootstrap_95_interval":np.quantile(v[valid],[.025,.975]).tolist() if valid.any() else None}


def correct_order(low_minus_high, sign):
    return (low_minus_high * sign > 0) + .5*(low_minus_high == 0)


def effects(cells):
    return 0.5 * (cells[...,2] - cells[...,0] + cells[...,3] - cells[...,1])


def contrasts(cells):
    return np.stack([cells[...,2]-cells[...,0], cells[...,3]-cells[...,1]],axis=-1)


def corr(a,b):
    a=a-a.mean(axis=(-2,-1),keepdims=True); b=b-b.mean(axis=(-2,-1),keepdims=True)
    num=(a*b).mean(axis=(-2,-1)); den=np.sqrt((a*a).mean(axis=(-2,-1))*(b*b).mean(axis=(-2,-1)))
    return np.divide(num,den,out=np.full_like(num,np.nan),where=den>1.e-15)


def rate(mask, positive, idx):
    num = (mask & positive).sum(axis=1)
    den = mask.sum(axis=1)
    boot = np.divide(num[idx].sum(axis=1),den[idx].sum(axis=1),
                     out=np.full(len(idx),np.nan),where=den[idx].sum(axis=1)>0)
    return {**with_interval(num.sum()/den.sum(),boot), "numerator":int(num.sum()), "denominator":int(den.sum())}


def score_endpoint(rows, endpoint, idx):
    rows = rows.sort_values(["family","condition"])
    if list(rows["condition"]) != list("abcd") * len(idx[0]):
        raise ValueError("Quartet alignment changed")
    gold = rows[ENDPOINTS[endpoint]].to_numpy().reshape(-1,4)
    y_effect = effects(gold)
    direction = -1 if endpoint=="coherence" else 1
    bootgold = gold[idx]
    result = {"n_families":len(gold), "observed_low_minus_high":with_interval(y_effect.mean(),y_effect[idx].mean(axis=1)),
              "observed_pair_direction":with_interval(correct_order(contrasts(gold),direction).mean(),
                                                      correct_order(contrasts(gold),direction)[idx].mean(axis=(1,2))),
              "models":{}, "difference_in_rmse_integration_minus_comparator":{}}
    raw = rows.raw_transition.to_numpy().reshape(-1,4)
    raw_correct = correct_order(contrasts(raw),1)
    result["uncalibrated_display_ordering_higher_for_low_coherence"] = with_interval(raw_correct.mean(),raw_correct[idx].mean(axis=(1,2)))
    rmse = {}
    for name in ARMS:
        pred = rows[name].to_numpy().reshape(-1,4)
        squared = ((pred-gold)**2).mean(axis=1)
        point, boot = np.sqrt(squared.mean()), np.sqrt(squared[idx].mean(axis=1))
        rmse[name] = (point, boot)
        d = effects(pred)
        mag_err = d-y_effect
        out = {"rmse":with_interval(point,boot), "low_minus_high":with_interval(d.mean(),d[idx].mean(axis=1)),
               "low_minus_high_error":with_interval(mag_err.mean(),mag_err[idx].mean(axis=1)),
               "itemwise_effect_mae":with_interval(abs(mag_err).mean(),abs(mag_err[idx]).mean(axis=1)),
               "correlation_with_observed":with_interval(corr(pred,gold),corr(pred[idx],bootgold)),
               "condition_pair_ordering":with_interval(correct_order(contrasts(pred),direction).mean(),
                                                       correct_order(contrasts(pred),direction)[idx].mean(axis=(1,2)))}
        if endpoint=="coherence":
            high=np.tile(np.array([True,True,False,False]),(len(gold),1))
            alarm = pred<3.
            out["midpoint_alerts"]={"false_alarm_in_experimental_high":rate(high,alarm,idx),
                                    "sensitivity_in_experimental_low":rate(~high,alarm,idx),
                                    "false_alarm_for_human_mean_at_least_3":rate(gold>=3,alarm,idx),
                                    "miss_for_human_mean_below_3":rate(gold<3,~alarm,idx)}
        result["models"][name]=out
    for name, (point,boot) in rmse.items():
        if name!="integration":
            result["difference_in_rmse_integration_minus_comparator"][name] = with_interval(
                rmse["integration"][0]-point,rmse["integration"][1]-boot)
    return result


def prefix_profile(rows, pre, matrices, fits):
    test = rows[rows.split.eq("holdout")]
    person = pre[pre.split.eq("holdout")].sort_values(["id","participant","word_index"]).copy()
    person["avg_to_word"] = person.groupby(["id","participant"]).log_ms.cumsum()/(person.word_index+1)
    human = person.groupby(["id","word_index"]).avg_to_word.mean()
    records=[]
    for k in range(1,int(test.critical_index.max())+1):
        p=test[test.critical_index.ge(k)].copy()
        if p.empty:
            continue
        x=np.stack([matrices[i][k-1] for i in p.id])
        y=np.array([human.loc[i,k-1] for i in p.id])
        v=apply(x, fits["integration"],"reading_time")
        norm=correct_order(contrasts(v.reshape(-1,4)),1)
        d=effects(y.reshape(-1,4)); m=effects(v.reshape(-1,4))
        idx=np.random.default_rng(9271).integers(len(d),size=(10000,len(d)))
        records.append({"prefix_words":k,"families":len(d),"human_cumulative_log_rt_low_minus_high":with_interval(d.mean(),d[idx].mean(axis=1)),
                        "model_cumulative_log_rt_low_minus_high":with_interval(m.mean(),m[idx].mean(axis=1)),
                        "predicted_condition_ordering":float(norm.mean())})
    return records


def score(frame, pre, matrices):
    lock=json.loads((OUT/"prediction-lock.json").read_text())
    forecasts=OUT/"heldout-predictions.csv"
    ck=OUT/"readouts.json"
    if sha(forecasts)!=lock["forecast_sha256"] or sha(ck)!=lock["coefficient_sha256"]:
        raise ValueError("Forecasts changed after sealing")
    p=pd.read_csv(forecasts)
    test=frame[frame.split.eq("holdout")]
    e=p.merge(test[["id",*ENDPOINTS.values()]],on="id",validate="m:1")
    results={}
    for end in ENDPOINTS:
        data=e[e.endpoint.eq(end)]
        n=data.family.nunique()
        idx=np.random.default_rng(9271).integers(n,size=(10000,n))
        results[end]=score_endpoint(data,end,idx)
    fits=json.loads(ck.read_text())["coefficients"]
    results["prefix_profile"]=prefix_profile(frame,pre,matrices,fits["reading_time"])
    results["success_primary"] = all(results["coherence"]["difference_in_rmse_integration_minus_comparator"][n]
                                     ["family_bootstrap_95_interval"][1]<0 for n in ("lexical","display"))
    dump(OUT/"scores.json",results)
    e.to_csv(OUT/"heldout-scored.csv",index=False,float_format="%.15g")
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("action",choices=["fit","score"])
    p.add_argument("--features",type=Path,required=True)
    args=p.parse_args()
    frame,pre,matrices,source=setup(args.features)
    if args.action=="fit":
        print(fit_and_lock(frame,matrices,source,args.features))
    else:
        r=score(frame,pre,matrices)
        for end in ENDPOINTS:
            print(end, "observed low-high",r[end]["observed_low_minus_high"])
            for name, m in r[end]["models"].items():
                print(name,m["rmse"],m["condition_pair_ordering"],m["low_minus_high"])
