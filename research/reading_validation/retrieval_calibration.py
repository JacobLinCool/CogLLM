"""Train the first-report hazards from human paths and matched experimental contrasts."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from research.reading_validation.memory_data import BASE, E1, OUT as MEMORY, load_rows, sha
from research.reading_validation.memory_fit import counts, design


def first_log_odds(v, prefix):
    """Return log odds of reporting for the first time at the last row, and its gradient."""
    logits = prefix @ v
    logp = -np.logaddexp(0., -logits[-1]) - np.logaddexp(0.,logits[:-1]).sum()
    other = -np.expm1(logp)
    if not other>0:
        raise FloatingPointError("First-report probability saturated")
    gradient = (1-expit(logits[-1]))*prefix[-1] - expit(logits[:-1]) @ prefix[:-1]
    return logp - np.log(other), gradient / other


def word_pool(record):
    z = np.array(record["all_token_logits"],dtype=float)
    out, start, previous = [],0,0.
    for end in record["ends"]:
        s = z[start:end+1]
        if not len(s):
            raise ValueError("Empty intervening tokens")
        peak = float(logsumexp(s)-np.log(len(s)))
        out.append([s.mean(),np.quantile(s,.9),peak,peak-previous])
        start, previous = end+1,peak
    return np.asarray(out)


def fit(articles, measurements, features, paired):
    items = [(r,features[r["id"]],measurements[r["id"]]) for r in articles]
    # Equal expected likelihood contribution per original reader-trial in each article.
    raw = np.vstack([x[1:] for _,x,_ in items])
    weight = np.concatenate([risk[1:]/n for _,_,(risk,_,n) in items])
    targets = np.concatenate([np.divide(event[1:],risk[1:],out=np.zeros(len(event)-1),where=risk[1:]>0)
                              for _,_,(risk,event,n) in items])
    weight /= weight.sum()
    mu = np.average(raw,axis=0,weights=weight)
    scale = np.sqrt(np.average((raw-mu)**2,axis=0,weights=weight))
    scale[scale<1e-7] = 1.
    def normalize(x):
        return np.column_stack((np.ones(len(x)),(x-mu)/scale))
    x = normalize(raw)
    families = {}
    for row,feat,(_,event,n) in items:
        t = row["target"]-1
        families.setdefault(row["story"],{})[row["arm"]] = (normalize(feat[1:t+1]),event[t]/n)
    contrasts = []
    for pair in families.values():
        a, ya = pair["inconsistent"]
        b, yb = pair["consistent"]
        disagree = ya*(1-yb) + yb*(1-ya)
        if disagree>0:
            contrasts.append((a,b,ya*(1-yb)/disagree))
    def objective(v):
        z = x@v
        loss = weight @ (np.logaddexp(0,z) - targets*z)
        grad = x.T @ (weight*(expit(z)-targets))
        if paired:
            pair_loss = 0.
            pair_grad = np.zeros_like(v)
            for a,b,y in contrasts:
                la,ga = first_log_odds(v,a)
                lb,gb = first_log_odds(v,b)
                difference = la-lb
                pair_loss += np.logaddexp(0,difference)-y*difference
                pair_grad += (expit(difference)-y)*(ga-gb)
            loss = .5*(loss + pair_loss/len(contrasts))
            grad = .5*(grad + pair_grad/len(contrasts))
        loss += .01*(v[1:]@v[1:])
        grad[1:] += .02*v[1:]
        return loss,grad
    center = np.zeros(x.shape[1])
    center[0] = -3.
    result = minimize(objective,center,jac=True,method="L-BFGS-B",options={"ftol":2e-14,"gtol":2e-8,"maxiter":600})
    if not result.success or abs(result.jac).max()>2e-5:
        raise ArithmeticError(f"Conditional observation model did not converge: {result.message}")
    return dict(center=mu.tolist(),scale=scale.tolist(),coef=result.x.tolist(),loss=float(result.fun),
                max_gradient=float(abs(result.jac).max()),iterations=int(result.nit))


def predict(model,x):
    v = np.asarray(model["coef"])
    hz = expit(((x-model["center"])/model["scale"])@v[1:]+v[0])
    hz[0] = 0.
    return hz


def calibrate(archive, output, extra=None):
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((archive/"receipt.json").read_bytes())
    record = archive/"stories-frozen.json"
    if sha(record)!=receipt["files"][record.name]:
        raise ValueError("Changed frozen token evidence")
    docs = {r["id"]:r["sentences"] for r in load_rows(BASE/"memory/stories.jsonl")}
    metadata = json.loads((BASE/"memory/study-map.json").read_bytes())
    train = [r for r in metadata if r["study"]==1]
    observed = counts(load_rows(E1/"outcomes.jsonl"),train)
    evidence = json.loads(record.read_bytes())
    values = {"surface":{}, "terminal":{}, "word":{}}
    if extra is not None:
        secondary = json.loads(extra.read_bytes())
        if set(secondary)!=set(docs):
            raise ValueError("Need one retrieval trajectory for every original story")
        values["retrieval"] = {}
        values["surprisal"] = {}
        values["transfer"] = {}
    for k,sentences in docs.items():
        values["surface"][k] = design(sentences)
        values["terminal"][k] = design(sentences,evidence[k]["logits"])
        values["word"][k] = np.column_stack((values["terminal"][k],word_pool(evidence[k])))
        if extra is not None:
            values["surprisal"][k] = np.column_stack((values["surface"][k],secondary[k]["surprisal"]))
            values["retrieval"][k] = np.column_stack((values["word"][k],secondary[k]["surprisal"],secondary[k]["retrieval"]))
            values["transfer"][k] = np.column_stack((values["surprisal"][k],secondary[k]["retrieval"]))
    forecasts,models = {},{}
    prior_file=MEMORY/"forecast/forecasts.npz"
    with np.load(prior_file,allow_pickle=False) as prior:
        for k in docs:
            forecasts["strict.original."+k] = prior["strict.frozen."+k]
    for family in sorted({r["story"] for r in train}):
        selected = [r for r in train if r["story"]!=family]
        for name,block in values.items():
            for matched in (False,True):
                label=name+(".paired" if matched else ".path")
                model=fit(selected,observed,block,matched)
                models[f"{family}.{label}"]=model
                for r in metadata:
                    if r["story"]==family:
                        forecasts["strict."+label+"."+r["id"]] = predict(model,block[r["id"]])
    output.mkdir(parents=True)
    np.savez_compressed(output/"forecasts.npz",**forecasts)
    (output/"models.json").write_text(json.dumps(models,allow_nan=False)+"\n")
    seal=dict(old_forecast=sha(prior_file),e1_only=True,solver_fits=len(models),evidence=sha(record),
              extra=sha(extra) if extra else None,
              models=["original"]+[k+"."+z for k in values for z in ("path","paired")],
              files={n:sha(output/n) for n in ("forecasts.npz","models.json")})
    (output/"seal.json").write_text(json.dumps(seal,indent=2)+"\n")
    print(f"Sealed {len(models)} converged fits")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--extra",type=Path)
    args=parser.parse_args()
    calibrate(args.archive,args.output,args.extra)
