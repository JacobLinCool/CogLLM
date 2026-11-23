"""Score held-reader and held-article predictions for scientific support choices."""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from research.reading_validation.joint_data import digest
from research.reading_validation.local_help_fit import OUTCOMES, data
from research.reading_validation.local_help_inference import (
    DESIGNS,
    HelpReadout,
    design,
)
from research.reading_validation.local_help_prepare import OUT, code
from research.reading_validation.local_jargon import two_way


def evaluate(records, bank, target, endpoint, *, matched):
    methods = ["fixed_lexical", *DESIGNS]
    score = np.column_stack([bank[endpoint + "." + arm] for arm in methods])
    papers = sorted({r["paper"] for r in records})
    readers = sorted({r["reader"] for r in records})
    y = np.array([r["targets"][target] for r in records])
    groups = defaultdict(list)
    for i, row in enumerate(records):
        if y[i] >= 0:
            groups[papers.index(row["paper"]), readers.index(row["reader"])].append(i)
    slab = np.zeros((len(papers), len(readers), len(methods)))
    count = np.zeros((len(papers), len(readers)), dtype=int)
    for (a, b), inds in groups.items():
        wins = []
        for i, j in combinations(inds, 2):
            if y[i] == y[j] or (matched and records[i]["prior"] != records[j]["prior"]):
                continue
            wins.append((1 + np.sign(y[i] - y[j]) * np.sign(score[i] - score[j])) / 2)
        if wins:
            count[a, b] = len(wins)
            slab[a, b] = np.mean(wins, axis=0)
    eligible = count > 0
    point, crossed, fixed = two_way(slab, eligible)

    def summarize(x, y, z):
        return {
            "estimate": float(x),
            "reader_article_95": np.quantile(y, [0.025, 0.975]).tolist(),
            "conditional_article_95": np.quantile(z, [0.025, 0.975]).tolist(),
        }

    result = {
        k: summarize(point[i], crossed[:, i], fixed[:, i])
        for i, k in enumerate(methods)
    }
    for a, b in (
        ("article_prior", "lexical_prior"),
        ("article_prior", "lexical_deft_prior"),
        ("article", "lexical"),
        ("article_prior", "article"),
        ("lexical_prior", "lexical"),
        ("article_prior", "fixed_lexical"),
    ):
        i, j = methods.index(a), methods.index(b)
        result[a + " minus " + b] = summarize(
            point[i] - point[j],
            crossed[:, i] - crossed[:, j],
            fixed[:, i] - fixed[:, j],
        )
    passed = (
        result["article_prior"]["reader_article_95"][0] > 0.5
        and result["article_prior minus lexical_prior"]["reader_article_95"][0] > 0
    )
    return {
        "informative_pairs": int(count.sum()),
        "reader_article_cells": int(eligible.sum()),
        "articles": int(eligible.any(1).sum()),
        "readers": int(eligible.any(0).sum()),
        "concordance": result,
        "exceeds_chance_and_lexical_prior": passed,
    }


def main():
    lock = json.loads((OUT / "forecast-seal.json").read_bytes())
    if (
        digest(OUT / "forecasts.npz") != lock["forecast_sha256"]
        or digest(Path(__file__)) != lock["score_sha256"]
    ):
        raise ValueError("Held-out scores or scoring code changed")
    rec, _ = data()
    keys = [code("/".join((r["paper"], r["term"], r["reader"]))) for r in rec]
    scores = {}
    with np.load(OUT / "forecasts.npz", allow_pickle=False) as bank:
        if keys != bank["key"].tolist():
            raise ValueError("Response keys differ from sealed predictions")
        for i, ep in enumerate(OUTCOMES):
            scores[ep] = {
                k: evaluate(rec, bank, i, ep, matched=v)
                for k, v in [("same_prior", True), ("all_priors", False)]
            }
    # Check every exported coefficient vector on the complete outcome-free feature matrix.
    features = np.asarray([r["features"] for r in rec])
    before = np.asarray([r["prior"] for r in rec])
    model = HelpReadout()
    replay = {}
    for endpoint in OUTCOMES:
        for arm in DESIGNS:
            w = model.weights[endpoint][arm]
            x = design(features, before, arm)
            expected = ((x - w["center"]) / w["scale"]) @ w["coef"] + w["intercept"]
            actual = model.predict(
                features, arm=arm, endpoint=endpoint, familiarity_before=before
            )
            if not np.isfinite(actual["probability"]).all():
                raise ValueError("Invalid exported probability")
            replay[endpoint + "." + arm] = float(
                np.max(np.abs(expected - actual["log_odds"]))
            )
    result = {
        "forecasts_sha256": lock["forecast_sha256"],
        "reserve_scored": False,
        "available_records": len(rec),
        "heldout_comparisons": scores,
        "serving_replay": replay,
    }
    (OUT / "scores.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    for e, v in scores.items():
        s = v["same_prior"]
        print(
            e,
            s["informative_pairs"],
            s["articles"],
            {
                k: round(q["estimate"], 4)
                for k, q in s["concordance"].items()
                if "minus" not in k
            },
            s["exceeds_chance_and_lexical_prior"],
        )


if __name__ == "__main__":
    main()
