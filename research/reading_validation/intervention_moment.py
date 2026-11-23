"""Fit the full first-report distribution and randomized, within-story critical incidence."""

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

from research.reading_validation.memory_data import BASE, load_rows, sha
from research.reading_validation.memory_fit import counts
from research.reading_validation.memory_score import reveal, summarize
from research.reading_validation.premise_emission import OUT as PREVIOUS, design
from research.reading_validation.referent_features import setup
from research.reading_validation.taxonomic_monitor import Route, category_features, normalized
from research.reading_validation.taxonomic_pairs import Ontology, form, native_words, opportunities, source_links

OUT = Path("research/results/reading-validation-intervention-moment")
PLAN = Path(__file__).with_name("intervention-moment-protocol.md")


def measurements(route, sentences, bags):
    """Measure the best-supported referential binding and uncertainty available at sentence completion."""
    result = []
    for j, (words, bag) in enumerate(zip(sentences, bags, strict=True)):
        role = route.target(j + 1, len(words))
        if len(bag.origin):
            logits = (bag.rank - route.mu) / route.sd @ route.v + route.offset[bag.origin]
            weight = softmax(logits)
            location = [np.square(weight).sum(), np.bincount(bag.origin, weight).max(),
                        np.bincount(bag.word, weight).max(), logsumexp(logits) - logsumexp(route.offset[bag.origin])]
            mu, sd, beta, _ = route.models["directional"]
            individual = expit(normalized(category_features(bag, "directional"), mu, sd) @ beta)
            # The independently estimated lexical norms distinguish inclusion from thematic similarity.
            relation = np.c_[individual, bag.directional[:, 0], bag.directional[:, 1], bag.symmetric[:, 0]]
            binder = np.r_[weight @ relation, relation[np.argmax(weight)]]
        else:
            location, binder = np.zeros(4), np.zeros(8)
        result.append(np.r_[design(j, words), role, location, binder, role * binder])
    return np.asarray(result)


def incidence(x, slope, index):
    """First-report probability and its exact derivative at a registered training coordinate."""
    logits = x[1:index + 1] @ slope
    hazard = expit(logits)
    probability = np.exp(-np.logaddexp(0., logits[:-1]).sum() - np.logaddexp(0., -logits[-1]))
    gradient = probability * ((1 - hazard[-1]) * x[index] - hazard[:-1] @ x[1:index])
    return probability, gradient


def mean_rate(events, n):
    p = (events + .5) / (n + 1.)
    return p * (1 - p) / (n + 2.)


def estimate(rows, x, obs, use_moment):
    groups = {}
    all_x, risks, actual = [], [], []
    for r in rows:
        key = r["id"]
        risk, events, readers = obs[key]
        all_x.append(x[key][1:])
        risks.extend(risk[1:] / readers)
        actual.extend(events[1:] / readers)
        t = r["target"] - 1
        groups.setdefault(r["story"], {})[r["arm"]] = (key, t, events[t]/readers, mean_rate(events[t], readers))
    features = np.vstack(all_x)
    weights = np.asarray(risks)
    center = np.average(features, axis=0, weights=weights)
    scale = np.sqrt(np.average((features - center)**2, axis=0, weights=weights))
    scale[scale < 1e-7] = 1.
    def transform(raw):
        return np.column_stack((np.ones(len(raw)), (raw - center) / scale))
    matrix = transform(features)
    n = len(rows)
    at, y = weights / n, np.asarray(actual) / n
    pairs = []
    for group in groups.values():
        k,t,p,v = group["inconsistent"]
        l,u,q,w = group["neutral"]
        if t != u:
            raise ValueError("Within-item target coordinates differ.")
        pairs.append((transform(x[k]), transform(x[l]), t, p-q, v+w))
    variance_floor = np.median([p[-1] for p in pairs])
    def objective(beta):
        ell = matrix @ beta
        loss = at @ np.logaddexp(0., ell) - y @ ell + .02 * (beta[1:] @ beta[1:])
        grad = matrix.T @ (at * expit(ell) - y)
        grad[1:] += .04 * beta[1:]
        if use_moment:
            for right, left, t, observed, variance in pairs:
                inc, d_inc = incidence(right, beta, t)
                neu, d_neu = incidence(left, beta, t)
                difference = inc - neu - observed
                precision = 1 / (len(pairs) * max(variance_floor, variance))
                loss += .5 * precision * difference**2
                grad += precision * difference * (d_inc - d_neu)
        return loss, grad
    start = np.zeros(matrix.shape[1])
    start[0] = np.log(y.sum() / (at.sum() - y.sum()))
    direction = np.cos(np.arange(len(start)))
    direction /= np.linalg.norm(direction)
    step = 1e-5 * direction
    numerical = (objective(start + step)[0] - objective(start - step)[0]) / 2e-5
    analytic = objective(start)[1] @ direction
    if abs(analytic - numerical) > 1e-6:
        raise ArithmeticError("First-incidence intervention derivative disagrees with central difference.")
    opt = minimize(objective, start, jac=True, method="L-BFGS-B",
                   options={"gtol": 1e-8, "ftol": 1e-14, "maxiter": 800})
    if not opt.success or np.max(np.abs(opt.jac)) > 5e-6:
        raise ArithmeticError(f"Moment fit did not converge: {opt.message}, {max(abs(opt.jac))}")
    def predict(raw):
        hazard = expit(transform(raw) @ opt.x)
        hazard[0] = 0.  # The experiment forces a consistent key at the initial sentence.
        return hazard
    return predict, dict(center=center.tolist(), scale=scale.tolist(), beta=opt.x.tolist(),
                         objective=float(opt.fun), gradient=float(np.max(np.abs(opt.jac))),
                         derivative_check=float(abs(analytic-numerical)), source_variance_floor=float(variance_floor))


def fit(archive):
    setup()
    folder = OUT / "forecast"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "seal.json").exists():
        raise FileExistsError("Complete intervention forecasts already exist.")
    rows = json.loads((BASE / "memory/study-map.json").read_bytes())
    raw = {r["id"]: r["sentences"] for r in load_rows(BASE / "memory/stories.jsonl")}
    docs = {key: list(map(form, s)) for key, s in raw.items()}
    native = native_words(archive, raw)
    ontology = Ontology()
    bags = {k: opportunities(s, native[k], ontology) for k,s in raw.items()}
    source = [r for r in rows if r["study"] in (1,2)]
    e2 = [r for r in source if r["study"] == 2]
    observation = counts(reveal(2, rows), e2)
    links = source_links(source, raw)
    @lru_cache(None)
    def router(exclusions):
        return Route([r for r in source if r["story"] not in exclusions], docs, bags, links)
    estimates, history = {}, {}
    for excluded in sorted({r["story"] for r in e2}):
        inner = [r for r in e2 if r["story"] != excluded]
        outer = [r for r in rows if r["story"] == excluded and r["study"] == 3]
        design_rows = {}
        for r in inner:
            key = r["id"]
            rr = router(tuple(sorted((excluded, r["story"]))))
            design_rows[key] = measurements(rr, docs[key], bags[key])
        history[excluded] = {}
        for mode in ("trial", "moment"):
            apply, fit_info = estimate(inner, design_rows, observation, mode == "moment")
            history[excluded][mode] = fit_info
            for r in outer:
                key = r["id"]
                val = measurements(router((excluded,)), docs[key], bags[key])
                estimates["strict."+mode+"."+key] = apply(val)
        print("Froze intervention predictions for held story", excluded, flush=True)
    old = PREVIOUS / "forecast"
    receipt = json.loads((old / "seal.json").read_bytes())
    if sha(old / "hazards.npz") != receipt["files"]["hazards.npz"]:
        raise ValueError("Reference readout changed since the earlier comparison.")
    with np.load(old / "hazards.npz") as reference:
        for key in reference.files:
            if key.startswith(("strict.original.", "strict.directional.", "strict.native.", "strict.oracle.")):
                estimates[key] = reference[key]
    np.savez_compressed(folder / "hazards.npz", **estimates)
    (folder / "fits.json").write_text(json.dumps(history, indent=2) + "\n")
    seal = dict(protocol=sha(PLAN), archive=sha(archive / "receipt.json"), previous=sha(old / "seal.json"),
                files={f:sha(folder / f) for f in ("hazards.npz","fits.json")}, held_outcome_access=False)
    (folder / "seal.json").write_text(json.dumps(seal, indent=2) + "\n")


def grade():
    folder = OUT / "forecast"
    receipt = json.loads((folder / "seal.json").read_bytes())
    if any(sha(folder / k) != v for k,v in receipt["files"].items()):
        raise ValueError("Frozen intervention readouts changed.")
    metadata = json.loads((BASE / "memory/study-map.json").read_bytes())
    held = reveal(3, metadata)
    with np.load(folder / "hazards.npz") as h:
        methods = ("original", "directional", "native", "oracle", "trial", "moment")
        scores = summarize(h, held, 376, cohorts=("strict",), methods=methods, focus="moment", controls=methods[:-1])
        by_family = {}
        for family in sorted({r["story"] for r in held}):
            pairs = {}
            for arm in ("inconsistent","neutral"):
                local = [r for r in held if r["story"] == family and r["arm"] == arm]
                key = local[0]["id"]; t = local[0]["target"]-1
                denom = len(local)
                human = counts(local, [{"id":key}])[key][1][t]/denom
                pairs[arm] = {"human":human, **{m:float(h["strict."+m+"."+key][t] * np.prod(1-h["strict."+m+"."+key][:t])) for m in methods}}
            by_family[family] = {key:pairs["inconsistent"][key]-p for key,p in pairs["neutral"].items()}
        scores["family_effects"] = by_family
    (OUT / "scores.json").write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit","score"))
    parser.add_argument("archive", type=Path, nargs="?")
    args = parser.parse_args()
    fit(args.archive) if args.action == "fit" else grade()
