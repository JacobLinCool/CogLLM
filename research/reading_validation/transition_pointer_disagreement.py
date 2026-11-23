"""Compare uncertainty in the all-word pointer with the choices of two different readers."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_direct_data import display
from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_game_prepare import silver_docs
from research.reading_validation.transition_pointer_data import OUT, ROOT, extract
from research.reading_validation.transition_pointer_score import ARMS
from research.reading_validation.transition_pointer_selective import validated_predictions
from research.reading_validation.transition_score import dump, sha, with_interval


PROTOCOL = Path("research/reading_validation/transition-pointer-disagreement-protocol.md")
REGISTERED = "c0683251fecbb6926abf6a25bb20efe30d178b5c9720499fb13f540906be9e0e"
METHODS = ("constant", "recency", "lexical", "headwise")


def pair_outcomes(spans):
    regions, entities = Counter(), Counter()
    for a, b, n, e, _ in spans:
        regions[a, b] += n
        entities[e] += n
    voters = sum(entities.values())
    if voters < 4:
        raise ValueError("An eligible pronoun needs four distinct voters")
    return [1 - sum(n * (n - 1) for n in counts.values()) / (voters * (voters - 1))
            for counts in (regions, entities)]


def development(rows):
    originals = {r["id"]: r for r in map(json.loads, (ROOT / "game-native-stimuli.jsonl").read_text().splitlines())
                 if r["domain"] == "game" and r["fold"] == "dev"}
    out = {}
    for d, _, fold, tree, chains, els, words in silver_docs():
        if fold != "dev" or d not in originals:
            continue
        row = rows[d]
        text, coords, _ = display(tree, els, words)
        if text != row["text"] or text != originals[d]["text"]:
            raise ValueError("Silver display has changed")
        old = originals[d]["words"]
        indices = {tuple(s[:2]): j for j, s in enumerate(old)}
        kept = [indices[tuple(span)] for span in row["words"]]
        observed, _ = extract(row, tree, chains, els, words, old, coords, kept, set(), set())
        simple = {str(q): [s[:3] for s in response["spans"]] for q, response in observed.items()}
        if simple != row["clicks"]:
            raise ValueError("Rebuilt silver votes differ from the source pointer")
        out[d] = {str(q): r for q, r in observed.items()}
    if set(out) != set(originals):
        raise ValueError("Missing a source calibration text")
    return out


def signals(row, logits, q):
    forms = [row["text"][a:b].casefold() for a, b in row["words"][:q]]
    _, group = np.unique(forms, return_inverse=True)
    scores = []
    for name in METHODS[1:]:
        z = logits[ARMS.index(name) - 1].astype(float)
        mass = np.exp(z - z.max())
        mass /= mass.sum()
        pooled = np.bincount(group, weights=mass)
        scores.append((1 - mass @ mass, 1 - pooled @ pooled))
    return np.asarray(scores)


def collection(rows, targets, forecast):
    records = []
    for d, mentions in targets.items():
        for key, r in mentions.items():
            p = forecast[f"{d}/{key}"]
            outcome = pair_outcomes(r["spans"])
            records.append((d, rows[d]["group"], int(key), outcome, signals(rows[d], p, int(key)),
                            sum(s[2] for s in r["spans"])))
    return records


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


def fit(index, y, w):
    mean = w @ index
    scale = np.sqrt(w @ (index - mean) ** 2)
    x = (index - mean) / scale if scale > 1e-12 else np.zeros_like(index)
    X = np.column_stack((np.ones_like(x), x))
    prior = w @ y
    b = np.array([np.log(prior / (1 - prior)), 0.])
    penalty = np.diag([0., .01])

    def loss(t):
        z = X @ t
        return w @ (np.logaddexp(0., z) - y * z) + .5 * t @ penalty @ t

    for count in range(64):
        p = sigmoid(X @ b)
        grad = X.T @ (w * (p - y)) + penalty @ b
        if b[1] == 0 and abs(grad[0]) <= 1e-10 and grad[1] >= 0:
            grad[1] = 0
        if np.max(abs(grad)) <= 1e-10:
            return {"index_mean": float(mean), "scale": float(scale), "intercept": float(b[0]), "slope": float(b[1]),
                    "max_constrained_gradient": float(np.max(abs(grad))), "iterations": count}
        hess = (X.T * (w * p * (1 - p))) @ X + penalty
        step = np.linalg.solve(hess, grad)
        factor = 1.
        while factor > 1e-12:
            cand = b - factor * step
            if cand[1] >= 0 and loss(cand) <= loss(b) - 1e-4 * factor * grad @ step + 1e-14:
                b = cand
                break
            factor *= .5
        else:
            raise ValueError("Constrained calibration did not descend")
    raise ValueError("Source calibration failed stationarity")


def concordance(q, score):
    ia, ib = np.nonzero(q[:, None] > q[None, :] + 1e-12)
    if not len(ia):
        return None, 0
    return np.mean((score[ia] > score[ib]) + .5 * (score[ia] == score[ib]), axis=0), len(ia)


def summary(rows):
    work = defaultdict(list)
    for w, value in rows:
        work[w].append(value)
    ids = sorted(work)
    if len(ids) < 2:
        return {"works": len(ids), "texts": len(rows)}
    per = np.stack([np.mean(work[g], axis=0) for g in ids])
    _, _, draws = grouped_distribution(ids)
    mean = per.mean(0)
    boot = draws @ per / len(ids)
    arms = {m: with_interval(mean[i], boot[:, i]) for i, m in enumerate(METHODS)}
    paired = {}
    for m in ("constant", "recency", "lexical"):
        i = METHODS.index(m)
        paired[f"headwise_minus_{m}"] = with_interval(mean[3] - mean[i], boot[:, 3] - boot[:, i])
    return {"works": len(ids), "texts": len(rows), "methods": arms, "paired": paired}


def score(dev):
    source_size = Counter(r[0] for r in dev)
    weights = np.array([1 / source_size[r[0]] for r in dev])
    weights /= weights.sum()
    source_y = np.array([r[3] for r in dev])
    source_x = np.stack([r[4] for r in dev])
    models = {}
    for t, name in enumerate(("literal", "entity")):
        y = source_y[:, t]
        model = {m: fit(source_x[:, a, t], y, weights) for a, m in enumerate(METHODS[1:])}
        model["constant"] = float(weights @ y)
        models[name] = model
    return models


def evaluate(held, models):
    groups = defaultdict(list)
    for r in held:
        groups[r[0]].append(r)
    out = {}
    for t, name in enumerate(("literal", "entity")):
        accum = defaultdict(list)
        pairs = 0
        nonunanimous = 0
        for entries in groups.values():
            group = entries[0][1]
            y = np.array([r[3][t] for r in entries])
            nonunanimous += int((y > 0).sum())
            index = np.column_stack((np.ones(len(entries)), np.stack([r[4][:, t] for r in entries])))
            score = np.empty_like(index)
            score[:, 0] = models[name]["constant"]
            for i, key in enumerate(METHODS[1:], 1):
                mod = models[name][key]
                term = ((index[:, i] - mod["index_mean"]) / mod["scale"]) if mod["scale"] > 1e-12 else 0.
                score[:, i] = sigmoid(mod["intercept"] + mod["slope"] * term)
            actual = y[:, None]
            for label, v in (("expected_pair_brier", actual * (1 - score) ** 2 + (1 - actual) * score ** 2),
                             ("forecast_bias", score - actual)):
                accum[label].append((group, v.mean(0)))
            for label, v in (("raw_concordance", index), ("calibrated_concordance", score)):
                c, n = concordance(y, v)
                if c is not None:
                    accum[label].append((group, c))
                    if label == "raw_concordance":
                        pairs += n
        out[name] = {"queries": len(held), "clicks": sum(r[-1] for r in held), "nonunanimous_queries": nonunanimous,
                     "unequal_within_text_pairs": pairs, **{k: summary(v) for k, v in accum.items()}}
    return out


def run(path):
    if sha(PROTOCOL) != REGISTERED:
        raise ValueError("The uncertainty protocol has changed")
    seal = json.loads((OUT / "source-seal.json").read_text())
    p = ROOT / "pointer-stimuli.jsonl"
    if sha(p) != seal["input_sha256"]:
        raise ValueError("Human source inputs changed")
    rows = {r["id"]: r for r in map(json.loads, p.read_text().splitlines())}
    forecast, h = validated_predictions(path, seal, seal["protocol_sha256"])
    calibration = collection(rows, development(rows), forecast)
    models = score(calibration)
    lock = OUT / "pointer-disagreement-calibration.json"
    dump(lock, {"protocol_sha256": REGISTERED, "forecast_sha256": h, "development_queries": len(calibration),
         "development_texts": len({r[0] for r in calibration}), "models": models})
    gold = OUT / "external-clicks.json"
    if sha(gold) != seal["gold_sha256"]:
        raise ValueError("Held-out votes changed")
    test = collection(rows, json.loads(gold.read_text())["articles"], forecast)
    result = evaluate(test, models)
    dump(OUT / "pointer-disagreement.json", {"source_calibration_sha256": sha(lock), "gold_sha256": sha(gold),
         "distinct_eligible_players": seal["distinct_eligible_clickers"]["test"], "evaluations": result})
    print({n: r["raw_concordance"]["methods"]["headwise"] for n, r in result.items()}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    run(parser.parse_args().features)
