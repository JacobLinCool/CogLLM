"""Cross-fit observed first-pass regressions against independent next-word expectations."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from wordfreq import zipf_frequency

from research.backbone_transfer.extract import digest, read_json, write_json


ROOT = Path(".cache/cogllm/reading-validation")
OUT = Path("research/results/reading-validation-expectation-rereading")
PLAN = Path(__file__).with_name("expectation-rereading-protocol.md")
EYE = ROOT / "provo-eyetracking.csv"
WORDS = ROOT / "cloze-population/external-scored-word-rows.jsonl"
EYE_DIGEST = "38aedcb29bc9171009916eb2bcc2375729f104a2a1005c64a563da94b611b9e7"


def read_words(surprisal):
    lock = read_json(Path("research/results/reading-validation-cloze-population/word-export.json"))
    if digest(WORDS) != lock["file"] or digest(EYE) != EYE_DIGEST:
        raise ValueError("Word norms or the original eye-tracking release have changed.")
    raw = [json.loads(x) for x in WORDS.read_text().splitlines()]
    lookup = {(r["Text_ID"], r["target_ordinal_zero_based"]): r for r in raw}
    if len(lookup) != len(raw):
        raise ValueError("Duplicate original word coordinates.")
    items, counts = [], Counter()
    for (article, pos), curr in lookup.items():
        nxt = lookup.get((article, pos + 1))
        if (nxt is None or not curr["normative_denominator_verified"]
                or not nxt["normative_denominator_verified"]):
            continue
        counts["clean_adjacent_words"] += 1
        if nxt["preceding_chars"] != curr["target_chars"]:
            raise ValueError("Two successive human prefixes did not share the same word.")
        word = curr["shown_target"]
        previous = curr["preceding_print"]
        coords = (article, pos)
        prior = (article, pos-1)
        if coords not in surprisal or prior not in surprisal:
            counts["unverified_current_or_previous_native_surprisal"] += 1
            continue
        cur_cost, prev_cost = surprisal[coords], surprisal[prior]
        if (cur_cost["word"] != word or prev_cost["word"] != previous
                or cur_cost["start"] != curr["target_chars"][0] or cur_cost["end"] != curr["target_chars"][1]
                or prev_cost["start"] != curr["preceding_chars"][0] or prev_cost["end"] != curr["preceding_chars"][1]):
            raise ValueError("Native surprisal and reader norms disagree on a printed word.")
        x = {"item": curr["id"], "passage": int(article), "position": pos+1, "printed": word}
        for name, printed, cost in (("current", word, cur_cost), ("previous", previous, prev_cost)):
            form = re.sub(r"^\W+|\W+$", "", printed.casefold())
            x[name+"_length"] = np.log1p(len(form))
            x[name+"_frequency"] = zipf_frequency(form, "en")
            x[name+"_punctuation"] = float(bool(re.search(r"[^\w\s]$", printed)))
            x[name+"_surprisal"] = cost["nats"]
        x["log_position"] = np.log1p(pos)
        for name, key in (("model","model_category_disagreement"), ("human", "human_category_disagreement")):
            before, after = curr[key], nxt[key]
            if name == "model":
                before, after = before["context"], after["context"]
            x[name+"_before"] = before
            x[name+"_rise"] = max(0., after-before)
            x[name+"_fall"] = max(0., before-after)
        items.append(x)
    if counts["clean_adjacent_words"] != 1566:
        raise ValueError("The originally verified Provo update panel has changed.")
    return pd.DataFrame(items), counts


def participants(items):
    fields = ["Participant_ID","Word_Unique_ID","Text_ID","Word_Number","Word","IA_SKIP","IA_FIRST_FIX_PROGRESSIVE",
              "IA_REGRESSION_OUT","IA_FIRST_RUN_DWELL_TIME"]
    eye = pd.read_csv(EYE, usecols=fields, encoding="mac_roman", dtype={"Word_Unique_ID":str}, low_memory=False)
    merged = items.merge(eye, left_on="item", right_on="Word_Unique_ID", validate="one_to_many")
    if merged.duplicated(["item","Participant_ID"]).any() or not np.array_equal(merged["passage"], merged["Text_ID"]):
        raise ValueError("Participant or article key was ambiguously joined.")
    if not np.array_equal(merged["position"], merged["Word_Number"]):
        raise ValueError("A joined word has different ordinal positions.")
    exact = merged["printed"] == merged["Word"]
    inside = ((merged["IA_SKIP"] == 0) & (merged["IA_FIRST_FIX_PROGRESSIVE"] == 1)
              & (merged["IA_FIRST_RUN_DWELL_TIME"] > 0) & merged["IA_REGRESSION_OUT"].isin([0,1]))
    sample = merged.loc[exact & inside].copy()
    sample["log_first_pass"] = np.log(sample["IA_FIRST_RUN_DWELL_TIME"])
    audit = dict(aligned_subject_word_rows=len(merged), different_print=int((~exact).sum()),
                 first_pass_opportunities=len(sample), participants=sample["Participant_ID"].nunique(),
                 passages=sample["passage"].nunique(), printed_words=sample["item"].nunique(),
                 regression_out=int(sample["IA_REGRESSION_OUT"].sum()), original_rows=len(eye))
    return sample, audit


def folds(values, salt):
    members = sorted(set(values), key=lambda v: sha256(f"revisit-provo/{salt}/{v}".encode()).digest())
    assigned = {v:i % 5 for i,v in enumerate(members)}
    return np.array([assigned[v] for v in values])


def fit(X, y, count):
    mass = count / count.sum()
    avg = mass @ X
    sd = np.sqrt(mass @ ((X-avg)**2))
    sd[sd < 1e-7] = 1.
    x = np.column_stack([np.ones(len(X)), (X-avg)/sd])
    def objective(b):
        z = x @ b
        value = mass @ (np.logaddexp(0,z) - y*z) + .0005*(b[1:]@b[1:])
        gradient = x.T@(mass*(expit(z)-y))
        gradient[1:] += .001*b[1:]
        return value, gradient
    res = minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B", jac=True,
                   options=dict(ftol=1.e-13, gtol=5.e-7, maxiter=500))
    if not res.success or np.max(np.abs(res.jac)) > 5e-6:
        raise ArithmeticError(f"A source-only rereading fit failed to converge: {res.message}")
    return avg, sd, res.x, float(np.max(np.abs(res.jac)))


def estimates(data):
    basic = [f"{period}_{var}" for period in ("current","previous")
             for var in ("length","frequency","punctuation","surprisal")] + ["log_position"]
    for col in ("current_length","previous_length","current_surprisal","previous_surprisal","log_position"):
        # Squares supply curvature on the original scale; source-only standardization follows.
        data[col+"_sq"] = data[col]**2
        basic.append(col+"_sq")
    response = data["IA_REGRESSION_OUT"].to_numpy(float)
    names = ("base","model","human")
    article_fold = folds(data["passage"], "article")
    reader_fold = folds(data["Participant_ID"], "person")
    output, fits = {}, {}
    for duration in (False, True):
        for arm in names:
            cols = basic + (["log_first_pass"] if duration else [])
            if arm != "base":
                cols = cols+[arm+"_"+key for key in ("before","rise","fall")]
            key = ("with_dwell/" if duration else "text/")+arm
            X = data[cols].to_numpy(float)
            if not np.isfinite(X).all():
                raise ValueError("A registered predictor has missing or nonfinite values.")
            pred = np.full(len(data),np.nan)
            fits[key] = []
            for article in range(5):
                for reader in range(5):
                    keep = (article_fold!=article)&(reader_fold!=reader)
                    test = (article_fold==article)&(reader_fold==reader)
                    if duration:
                        cov,y,n = X[keep],response[keep],np.ones(keep.sum())
                    else:
                        # Exactly the individual Bernoulli likelihood for repeated word rows.
                        subset = data.loc[keep]
                        group = subset.groupby("item", sort=True)
                        cov = group[cols].first().to_numpy()
                        y = group["IA_REGRESSION_OUT"].mean().to_numpy()
                        n = group.size().to_numpy()
                    mu,sd,b,grad = fit(cov,y,n)
                    pred[test] = ((X[test]-mu)/sd)@b[1:] + b[0]
                    fits[key].append(dict(article_fold=article,reader_fold=reader,source_rows=int(keep.sum()),
                                          gradient=grad,coefficient=dict(zip(cols,b[1:].tolist()))))
            if not np.isfinite(pred).all():
                raise ValueError("A held observation was omitted.")
            output[key] = pred
    return response, output, fits


def evaluate(data, response, forecasts):
    _,people = np.unique(data["Participant_ID"],return_inverse=True)
    _,texts = np.unique(data["passage"],return_inverse=True)
    n,k = people.max()+1, texts.max()+1
    idx = people*k+texts
    def block(v):
        return np.bincount(idx,weights=v,minlength=n*k).reshape(n,k)
    rng = np.random.default_rng(23750)
    readers = np.vstack((np.ones(n),rng.multinomial(n,np.full(n,1/n),size=4000)))
    articles = np.vstack((np.ones(k),rng.multinomial(k,np.full(k,1/k),size=4000)))
    def crossed(table):
        return np.sum((readers@table)*articles,axis=1)
    denom = crossed(block(np.ones(len(data))))
    risks,summary = {},{}
    for key,z in forecasts.items():
        nll = crossed(block(np.logaddexp(0,z)-response*z))/denom
        brier = crossed(block((expit(z)-response)**2))/denom
        risks[key] = nll
        summary[key] = dict(nll=nll[0],brier=brier[0],
                            interval95=np.quantile(nll[1:],[.025,.975]).tolist())
    comparisons = {}
    for mode in ("text","with_dwell"):
        dis = np.column_stack([risks[mode+"/base"]-risks[mode+"/"+a] for a in ("model","human")])
        scale = dis[1:].std(axis=0,ddof=1)
        critical = np.quantile(np.max(np.abs((dis[1:]-dis[0])/scale),axis=1),.95)
        for i,arm in enumerate(("model","human")):
            margin = scale[i]*critical
            comparisons[mode+"/"+arm] = dict(nll_reduction=dis[0,i],simultaneous95=[dis[0,i]-margin,dis[0,i]+margin],
                                               pointwise95=np.quantile(dis[1:,i],[.025,.975]).tolist())
    return summary, comparisons


def run(native):
    receipt=read_json(Path("research/results/reading-validation-cloze-population/native-serving-verification.json"))
    if digest(native)!=receipt["genuine_wholeword_surprisal_file"]:
        raise ValueError("The native whole-word costs failed their accelerator receipt.")
    geometry={r["id"]:r for r in map(json.loads,WORDS.read_text().splitlines())}
    with native.open() as file:
        records = [json.loads(row) for row in file]
    cost={}
    for record in records:
        g=geometry[record["id"]]
        coord=(record["Text_ID"],record["target_word"])
        if (coord!=(g["Text_ID"],g["target_ordinal_zero_based"]) or record["uid"]!=g["entity"]
                or record["word"]!=g["shown_target"] or coord in cost):
            raise ValueError("A certified native readout does not match its original text and word.")
        a,b=g["target_chars"]
        cost[coord]=dict(word=record["word"],start=a,end=b,nats=record["native_observed_whole_word_surprisal"])
    if len(records)!=len(geometry) or any(not np.isfinite(r["nats"]) or r["nats"] < 0 for r in cost.values()):
        raise ValueError("Surprisal input must contain unique, verified printed words.")
    words, audit = read_words(cost)
    sample, eye_audit = participants(words)
    y, forecasts, fits = estimates(sample)
    results, paired = evaluate(sample,y,forecasts)
    OUT.mkdir(exist_ok=True,parents=True)
    archive=OUT/"predictions.npz"
    np.savez_compressed(archive, **forecasts, observed=y,
                        participant=sample["Participant_ID"].to_numpy(str),word=sample["item"].to_numpy(str))
    write_json(OUT/"scores.json", dict(protocol=digest(PLAN), original_eye=digest(EYE),cloze=digest(WORDS),
               native=digest(native),forecast_sha256=digest(archive),norm_audit=dict(audit),eye_audit=eye_audit,
               estimates=results,paired=paired,fits=fits))
    print(eye_audit,paired)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("verified_word_surprisals",type=Path,help="Native Provo word-cost export from the certified L4 replay")
    run(p.parse_args().verified_word_surprisals)
