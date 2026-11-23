"""Bound the source-selected character screen under all original movie span placements."""
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np

from research.reading_validation.narrative_inference import DECISIONS_SHA, MarginalNarrativeInference
from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_full_movie_bounds import construct
from research.reading_validation.transition_partial_loss import OUT
from research.reading_validation.transition_score import dump, sha
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_prepare import RESULTS
from research.reading_validation.transition_writer_export import SOURCE


METRICS = ("precision", "recall", "f1", "false_flags_per_1000_words")
PROTOCOL = Path("research/reading_validation/transition-balanced-word-protocol.md")


def counts_by_segment(segments, flags_a, flags_b):
    configurations = []
    for ix, mask in segments:
        counts = np.stack((mask.sum(1), mask @ flags_a[ix].astype(int), mask @ flags_b[ix].astype(int)), axis=1)
        configurations.append(np.unique(counts, axis=0))
    return configurations


def weights(metric, true_words, flagged, words):
    if metric == "precision":
        return (1 / flagged if flagged else 0), 0.
    if metric == "recall":
        return (1 / true_words if true_words else 0), 0.
    if metric == "f1":
        denom = true_words + flagged
        return (2 / denom if denom else 0), 0.
    if metric == "false_flags_per_1000_words":
        return -1000 / words, 1000 * flagged / words
    raise ValueError(metric)


def sharp_bounds(configs, counts, words):
    fixed = np.zeros(3, dtype=int)
    variable = []
    for row in configs:
        if len(row) == 1:
            fixed += row[0]
        else:
            variable.append(row)
    counts = list(map(int, counts))
    low = int(fixed[0] + sum(c[:, 0].min() for c in variable))
    high = int(fixed[0] + sum(c[:, 0].max() for c in variable))
    keys = [(method, metric) for method in ("served_marginal", "ner_surprisal", "paired") for metric in METRICS]
    lower = np.full(len(keys), np.inf); upper = -lower
    possible = []
    for p in range(low, high + 1):
        w = []; offset = []
        for method, metric in keys:
            a, oa = weights(metric, p, counts[0], words)
            b, ob = weights(metric, p, counts[1], words)
            if method == "served_marginal":
                b, ob = 0., 0.
            elif method == "ner_surprisal":
                a, oa = 0., 0.
            else:
                b, ob = -b, -ob
            w.append([a, b]); offset.append(oa + ob)
        w = np.array(w)
        initial = w @ fixed[1:] + offset
        states = {int(fixed[0]): (initial, initial)}
        for choices in variable:
            updated = {}
            for n, (lo, hi) in states.items():
                for extra, ta, tb in choices:
                    new = int(n + extra)
                    if new > p:
                        continue
                    change = w[:, 0] * ta + w[:, 1] * tb
                    l, h = lo + change, hi + change
                    if new in updated:
                        l = np.minimum(l, updated[new][0]); h = np.maximum(h, updated[new][1])
                    updated[new] = l, h
            states = updated
        if p in states:
            lower = np.minimum(lower, states[p][0]); upper = np.maximum(upper, states[p][1]); possible.append(p)
    if not possible:
        raise ValueError("No feasible human labeling")
    scores = defaultdict(dict)
    for i, (method, metric) in enumerate(keys):
        scores[method][metric] = [float(lower[i]), float(upper[i])]
    return {"words": words, "model_and_control_flags": counts, "possible_positive_totals": possible, "scores": dict(scores)}


def verify():
    configs = [np.array([[1, 0, 1], [2, 1, 0]]), np.array([[0, 0, 0], [1, 1, 1], [2, 2, 1]])]
    counts = [3, 2]; population = 8
    res = sharp_bounds(configs, counts, population)
    for metric in METRICS:
        paired = []
        for choices in product(*configs):
            p, ta, tb = np.sum(choices, axis=0)
            a, oa = weights(metric, p, counts[0], population); b, ob = weights(metric, p, counts[1], population)
            paired.append(a * ta + oa - b * tb - ob)
        np.testing.assert_allclose(res["scores"]["paired"][metric], [min(paired),max(paired)], atol=1e-12)


def main():
    verify()
    segments, audit, names = construct()
    docs, _ = inputs()
    cut = json.loads(SOURCE.read_text())["model_decisions"]["words"]["ner_surprisal"]["cutoff"]
    qpath = OUT / "locked-forecasts.npz"; npath = RESULTS / "entity-control-probabilities.npz"
    if (sha(qpath) != json.loads((OUT / "scores.json").read_text())["forecasts_sha256"]
            or sha(npath) != json.loads((RESULTS / "entity-control-lock.json").read_text())["forecast_sha256"]):
        raise ValueError("The human-blind probability files changed")
    with np.load(qpath) as fit, np.load(npath) as ner:
        p = np.exp(-np.logaddexp(0., -fit["set_marginal"]))
        control = np.concatenate([ner[name][:, 0] for name in names]) >= cut
    model = MarginalNarrativeInference.for_writers().flags(p)
    configs = counts_by_segment(segments, model, control)
    total = sharp_bounds(configs, [model.sum(), control.sum()], len(p))
    ends = np.cumsum([len(docs[n]["words"]) for n in names])
    per_title = {}
    for i, (n, b) in enumerate(zip(names, ends, strict=True)):
        a = int(ends[i-1]) if i else 0
        subset = [c for (ix,_), c in zip(segments, configs, strict=True) if a <= ix[0] < b]
        per_title[n] = sharp_bounds(subset, [model[a:b].sum(),control[a:b].sum()], int(b-a))
    titles, _, draws = grouped_distribution(names)
    macro = {}
    for metric in METRICS:
        values = np.array([per_title[n]["scores"]["paired"][metric] for n in titles])
        boot = draws @ values / len(titles)
        macro[metric] = {"sharp_macro_range":values.mean(0).tolist(),
                         "whole_movie_sampling_and_alignment_envelope":[float(np.quantile(boot[:,0], .025)),float(np.quantile(boot[:,1], .975))],
                         "positive_for_every_placement":int((values[:,0] > 1e-10).sum()),
                         "negative_for_every_placement":int((values[:,1] < -1e-10).sum())}
    result = {"protocol_sha256":sha(PROTOCOL),"source_decisions_sha256":sha(SOURCE),"writer_policy_sha256":DECISIONS_SHA,
              "word_forecasts_sha256":[sha(qpath),sha(npath)],"measurement_audit":dict(audit),"pooled":total,
              "titles":len(names),"macro_title_paired":macro,"by_title":per_title}
    dump(OUT / "serving/balanced-all-words.json", result)
    print(total, macro, flush=True)


if __name__ == "__main__":
    main()
