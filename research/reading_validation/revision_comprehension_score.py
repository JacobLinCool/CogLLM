"""Evaluate sealed passage-level forecasts on individual reading-comprehension responses."""

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from research.reading_validation.joint_data import digest, lines
from research.reading_validation.revision_comprehension_prepare import (
    RAW,
    RAW_HASH,
    RESULT,
)

SEED = 486130
DRAWS = 10000
CONTROLS = ("source_word_recall", "edit_meaning", "retained_length")


def indicator(value):
    choices = {"0": 0, "0.0": 0, "false": 0, "1": 1, "1.0": 1, "true": 1}
    return choices[value.strip().casefold()]


def read_responses():
    if digest(RAW) != RAW_HASH:
        raise ValueError("Changed participant-response table")
    response = defaultdict(list)
    with RAW.open(newline="", encoding="utf8") as stream:
        for r in csv.DictReader(stream):
            correct = [indicator(r[f"is_correct_answer_{i}"]) for i in (1, 2, 3)]
            missing = [indicator(r[f"non_answer_{i}"]) for i in (1, 2, 3)]
            if sum(correct) != int(float(r["correct_count"])) or any(
                x and y for x, y in zip(correct, missing)
            ):
                raise ValueError(
                    "Question correctness and unanswerability columns disagree"
                )
            response[r["article_id"], r["para_id"], r["model_name"]].append(
                np.column_stack([correct, 1 - np.array(missing)])
            )
    return response


def estimate(points, outcomes, included, original, *, allow_repeats=False):
    lookup = defaultdict(dict)
    methods = sorted(next(iter(points.values())))
    for (article, para, name), vals in outcomes.items():
        key = article, para, name
        if article in included and (len(vals) == 1 or allow_repeats):
            human = np.mean(vals, axis=0)
            scores = np.array([points[key][m] for m in methods])
            lookup[article, para][name] = (human, scores)

    result = {}
    for k, endpoint in enumerate(("correctness", "answerability")):
        per_article = defaultdict(list)
        comparisons = question_count = paragraphs = 0
        for (article, _), info in sorted(lookup.items()):
            versions = sorted(set(info) - {"Adv"})
            pairs = (
                [("Adv", v) for v in versions]
                if original
                else list(combinations(versions, 2))
            )
            questions = []
            for i in range(3):
                wins = []
                for left, right in pairs:
                    h, f = info[left]
                    g, s = info[right]
                    different = np.sign(h[i, k] - g[i, k])
                    if different:
                        wins.append(0.5 + 0.5 * different * np.sign(f - s))
                if wins:
                    comparisons += len(wins)
                    questions.append(np.mean(wins, axis=0))
            if questions:
                question_count += len(questions)
                paragraphs += 1
                per_article[article].append(np.mean(questions, axis=0))
        if not per_article:
            result[endpoint] = {"articles": 0, "observed_pairs": 0}
            continue
        group_values = np.stack(
            [np.mean(v, axis=0) for _, v in sorted(per_article.items())]
        )
        n = len(group_values)
        mean = group_values.mean(axis=0)
        weights = (
            np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=DRAWS)
            / n
        )
        sampled = weights @ group_values

        def summary(point, values):
            return {
                "estimate": float(point),
                "article_bootstrap_95": np.quantile(values, [0.025, 0.975]).tolist(),
            }

        values = {m: summary(mean[j], sampled[:, j]) for j, m in enumerate(methods)}
        primary = methods.index("meaning")
        contrasts = {
            "meaning minus " + m: summary(
                mean[primary] - mean[j], sampled[:, primary] - sampled[:, j]
            )
            for j, m in enumerate(methods)
            if m != "meaning"
        }
        gate = all(
            contrasts["meaning minus " + m]["article_bootstrap_95"][0] > 0
            for m in CONTROLS
        )
        result[endpoint] = {
            "articles": n,
            "paragraphs": paragraphs,
            "questions": question_count,
            "observed_pairs": comparisons,
            "model": values,
            "paired": contrasts,
            "exceeds_all_primary_controls": gate,
        }
    return result


def simulation():
    methods = ("meaning", *CONTROLS)
    scores = {}
    observations = {}
    for a in ("x", "y"):
        for name, value in (("Adv", 1), ("low", 0), ("high", 1), ("same", 1)):
            key = a, "p", name
            scores[key] = dict.fromkeys(methods, 1 - value)
            scores[key]["meaning"] = value
            observations[key] = [np.full((3, 2), value)]
    for original, pairs in ((False, 12), (True, 6)):
        r = estimate(scores, observations, {"x", "y"}, original)["correctness"]
        assert r["observed_pairs"] == pairs and r["model"]["meaning"]["estimate"] == 1
        assert (
            all(r["model"][c]["estimate"] == 0 for c in CONTROLS)
            and r["exceeds_all_primary_controls"]
        )
    return {
        "opposed_predictions": "passed",
        "original_and_revision_pair_denominators": "passed",
    }


def main():
    input_seal = json.loads((RESULT / "input-seal.json").read_bytes())
    forecast = json.loads((RESULT / "forecast-seal.json").read_bytes())
    if (
        digest(RESULT / "forecasts.jsonl") != forecast["forecasts_sha256"]
        or digest(Path(__file__)) != forecast["frozen_score_script_sha256"]
    ):
        raise ValueError("Predictions or the frozen score calculation changed")
    if digest(RESULT / "input-seal.json") != forecast["geometry_sha256"]:
        raise ValueError("Geometry seal changed")
    scored = {
        (r["article"], r["paragraph"], r["condition"]): r["scores"]
        for r in lines(RESULT / "forecasts.jsonl")
    }
    simulation_check = simulation()
    observations = read_responses()
    if scored.keys() != observations.keys():
        raise ValueError("Incomplete prediction join")
    all_articles = {k[0] for k in scored}
    excluded = set(input_seal["excluded_from_primary"])
    clean = all_articles - excluded
    if not clean:
        raise ValueError("No disjoint articles for the primary evaluation")
    result = {
        "forecast_seal_sha256": digest(RESULT / "forecast-seal.json"),
        "simulation": simulation_check,
        "draws": DRAWS,
        "seed": SEED,
        "observed_presentations": sum(map(len, observations.values())),
        "response_rows_with_ambiguous_identity": sum(
            len(v) for v in observations.values() if len(v) > 1
        ),
        "primary": estimate(scored, observations, clean, False),
        "original_versus_revision": estimate(scored, observations, clean, True),
        "including_repeat_means": estimate(
            scored, observations, clean, False, allow_repeats=True
        ),
    }
    if excluded:
        result["all_including_supervised_overlap"] = estimate(
            scored, observations, all_articles, False
        )
    else:
        result["all_including_supervised_overlap"] = "Identical to the primary sample."
    (RESULT / "scores.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    for name, r in result["primary"].items():
        print(
            name,
            r["questions"],
            r["observed_pairs"],
            {k: round(v["estimate"], 3) for k, v in r["model"].items()},
            r["exceeds_all_primary_controls"],
        )


if __name__ == "__main__":
    main()
