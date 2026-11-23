"""Propagate pronoun-to-pronoun links to terminal earlier words in the same article forward."""

from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_pointer_data import OUT, ROOT
from research.reading_validation.transition_pointer_gap_data import OUT as GAP
from research.reading_validation.transition_pointer_score import ARMS, hit_k
from research.reading_validation.transition_pointer_selective import validated_predictions
from research.reading_validation.transition_score import dump, sha, with_interval


PROTOCOL = Path("research/reading_validation/transition-pointer-chain-protocol.md")
REGISTERED = "4a76a556e12c926ffdaddded592e580db21e6c565cc4b1a948880f26e1e439d9"
METHODS = ("lexical", "headwise")
READINGS = ("lexical_raw", "lexical_terminal", "headwise_raw", "headwise_terminal")


def distribute(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def propagate(raw_by_query):
    """Each eligible prior pronoun transfers its original edge weight exactly once."""
    readings = {}
    for q, raw in sorted(raw_by_query.items()):
        values = np.array(raw, copy=True, dtype=np.float64)
        if values.shape != (len(METHODS), q) or np.any(values < 0):
            raise ValueError("Pointer has unobserved or negative mass")
        for i, (_, earlier) in readings.items():
            values[:, :i] += raw[:, i, None] * earlier
            values[:, i] = 0
        if not np.isfinite(values).all() or not np.allclose(values.sum(1), 1, rtol=0, atol=2e-12):
            raise ValueError("Earlier-word propagation lost probability")
        readings[q] = raw, values
    return {q: np.stack([raw[0], resolved[0], raw[1], resolved[1]]) for q, (raw, resolved) in readings.items()}


def article_readings(row, archive):
    indices = [ARMS.index(m) - 1 for m in METHODS]
    raw = {q: distribute(archive[f"{row['id']}/{q}"][indices].astype(float)) for q in row["queries"]}
    return propagate(raw)


def aggregate(items, metric_names):
    groups = defaultdict(list)
    for group, values, _ in items:
        groups[group].append(values)
    ids = sorted(groups)
    if len(ids) < 2:
        return {"groups": len(ids), "queries": len(items)}
    matrix = np.stack([np.stack(groups[g]).mean(0) for g in ids])
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite chain evaluation requires an explicit region-availability audit")
    _, _, draws = grouped_distribution(ids)
    boot = (draws @ matrix.reshape(len(ids), -1)).reshape(len(draws), *matrix.shape[1:]) / len(ids)
    mean = matrix.mean(0)
    per = {arm: {label: with_interval(mean[i, k], boot[:, i, k]) for k, label in enumerate(metric_names)}
           for i, arm in enumerate(READINGS)}
    differences = {}
    for left, right in ((1, 0), (3, 2), (3, 1), (2, 0)):
        diff = mean[left] - mean[right]
        sampled = boot[:, left] - boot[:, right]
        differences[f"{READINGS[left]}_minus_{READINGS[right]}"] = {
            label: with_interval(diff[k], sampled[:, k]) for k, label in enumerate(metric_names)}
    return {"groups": len(ids), "queries": len(items), "responses": sum(n for _, _, n in items),
            "methods": per, "paired": differences}


def game_scores(row, labels, data, items):
    for key, label in labels.items():
        query = int(key)
        probs = data[query]
        forecasts, votes = [], []
        for first, stop, n, entity, _ in label["spans"]:
            literal = np.zeros(query, bool)
            literal[first:stop] = True
            equiv = np.zeros(query, bool)
            for a, b in label["equivalent"][entity]:
                equiv[a:b] = True
            metrics = []
            for weights in probs:
                metrics.append([hit_k(weights, literal, 1), hit_k(weights, literal, 3),
                                hit_k(weights, equiv, 1), hit_k(weights, equiv, 3), weights[equiv].sum(),
                                weights[literal].sum(), float(weights[literal].sum() == 0)])
            forecasts.append(metrics)
            votes.append(n)
        values = np.average(np.array(forecasts), axis=0, weights=votes)
        items["all"].append((row["id"], values, sum(votes)))
        items["distinct_works"].append((row["group"], values, sum(votes)))


def gap_scores(row, gold, data, items):
    for label in gold:
        query = label["q"]
        good, other = (label["B"], label["A"]) if label["target"] else (label["A"], label["B"])
        mask = np.zeros(query, bool)
        mask[good] = True
        metrics = []
        for weights in data[query]:
            target, distractor = weights[good].sum(), weights[other].sum()
            if target <= 0:
                raise ValueError("A human name contains only eliminated pronouns; record infinity explicitly before reporting")
            a, b = target / len(good), distractor / len(other)
            metrics.append([hit_k(weights, mask, 1), hit_k(weights, mask, 3), -np.log(target), target,
                            float(target > distractor) + .5 * (target == distractor), float(a > b) + .5 * (a == b)])
        labels = ["all", "intervening_pronoun" if any(min(good + other) < t < query for t in data) else "no_intervening_pronoun"]
        if max(good) < max(other):
            labels.append("farther_correct_name")
        if len(good) == len(other):
            labels.append("equal_name_lengths")
        for name in labels:
            items[name].append((label["article"], np.array(metrics), 1))


def score(kind, folder):
    if sha(PROTOCOL) != REGISTERED:
        raise ValueError("Terminal referent protocol has changed")
    is_game = kind == "game"
    dest = OUT if is_game else GAP
    seal = json.loads((dest / "source-seal.json").read_text())
    stimuli = ROOT / ("pointer-stimuli.jsonl" if is_game else "gap-pointer-stimuli.jsonl")
    gold_path = dest / ("external-clicks.json" if is_game else "external-regions.json")
    if sha(stimuli) != seal["input_sha256"] or sha(gold_path) != seal["gold_sha256"]:
        raise ValueError("External article or its annotations changed")
    pack, receipt = validated_predictions(folder, seal, seal["protocol_sha256"])
    metadata = json.loads(gold_path.read_text())
    if is_game:
        gold = metadata["articles"]
        columns = ("literal_top1", "literal_top3", "entity_top1", "entity_top3", "entity_mass", "literal_mass",
                   "literal_region_zero_probability")
    else:
        gold = defaultdict(list)
        for item in metadata["rows"]:
            gold[item["id"]].append(item)
        columns = ("open_top1", "open_top3", "name_nll", "name_mass", "two_name_correct", "length_adjusted_preference")
    items = defaultdict(list)
    for row in map(json.loads, stimuli.read_text().splitlines()):
        if not gold.get(row["id"]):
            continue
        readings = article_readings(row, pack)
        if is_game:
            game_scores(row, gold[row["id"]], readings, items)
        else:
            gap_scores(row, gold[row["id"]], readings, items)
    values = {name: aggregate(val, columns) for name, val in items.items()}
    dump(dest / "terminal-reference.json", {"protocol_sha256": REGISTERED, "forecast_sha256": receipt,
         "public_outcome_sha256": sha(gold_path), "evaluations": values})
    print(kind, values["all"]["paired"]["headwise_terminal_minus_headwise_raw"], flush=True)


def test():
    raw = {1: np.array([[1.], [1.]]), 3: np.array([[.2, .3, .5], [.5, .4, .1]]),
           4: np.array([[.1, .2, .3, .4], [.1, .2, .3, .4]])}
    resolved = propagate(raw)
    np.testing.assert_allclose(resolved[4][1], [.5, 0., .5, 0.])
    np.testing.assert_allclose(resolved[4][3], [.66, 0., .34, 0.])
    np.testing.assert_array_equal(resolved[3][0], raw[3][0])


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--kind", choices=("game", "gap"), required=True)
    args = parser.parse_args()
    test()
    score(args.kind, args.source)
