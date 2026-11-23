"""Score sealed CompLex predictions against the local held-out ratings."""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import json
import unicodedata

import numpy as np

from ..stage3.probe import _jsonl
from .complex_inputs import digest


METHODS = ("target_only", "target_end", "full_sentence")
FRACTIONS = (25, 50, 100)
SEEDS = tuple(range(5))
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 43
TIE_TOLERANCE = 1e-10


def _summary(values):
    return {"mean": float(np.mean(values)), "by_seed": values.tolist()}


def _table(array):
    return {
        method: {str(fraction): _summary(values) for fraction, values in zip(FRACTIONS, row, strict=True)}
        for method, row in zip(METHODS, array, strict=True)
    }


def _paired_delta(delta, groups):
    _, positions = np.unique(groups, return_inverse=True)
    group_counts = np.bincount(positions)
    group_losses = np.bincount(positions, weights=delta.mean(axis=0))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(group_counts), size=(BOOTSTRAP_DRAWS, len(group_counts)))
    means = group_losses[draws].sum(axis=1) / group_counts[draws].sum(axis=1)
    return {
        "difference": _summary(delta.mean(axis=-1)),
        "sentence_group_bootstrap_95_ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def lexical_key(row):
    return row["kind"], unicodedata.normalize("NFKC", row["target_text"]).casefold()


def _paired_context_ranking(rows, expected, fitted):
    same_targets = defaultdict(list)
    for index, row in enumerate(rows):
        same_targets[(row["kind"], row["target_text"])].append(index)
    left, right, target_count, human_ties = [], [], 0, 0
    for indices in same_targets.values():
        unequal = False
        for i, j in combinations(indices, 2):
            if rows[i]["context_group"] == rows[j]["context_group"]:
                continue
            if expected[i] == expected[j]:
                human_ties += 1
                continue
            left.append(i)
            right.append(j)
            unequal = True
        target_count += unequal
    if not left:
        raise ValueError("No same-target held-out contexts have distinct ratings")
    direction = np.sign(expected[left] - expected[right])
    differences = fitted[..., left] - fitted[..., right]
    if np.any(np.abs(differences[0]) > TIE_TOLERANCE):
        raise ValueError("Identical target-only inputs produced different estimates within a fit")
    concordance = np.where(np.abs(differences) <= TIE_TOLERANCE, 0.5, differences * direction > 0)
    return {
        "exact_surface_targets_with_eligible_pairs": target_count,
        "pairs_with_different_human_scores": len(left),
        "pairs_with_equal_human_scores": human_ties,
        "concordance": _table(concordance.mean(axis=-1)),
    }


def evaluate(design, observations, predictions, output):
    rows = list(_jsonl(design))
    trials = {row["id"]: row for row in rows if row["split"] == "test"}
    training = {lexical_key(row) for row in rows if row["split"] == "train"}
    if len(trials) != 1094 or any(row["value"] is not None for row in trials.values()):
        raise ValueError("Expected the complete held-out target inventory without values")
    originals = {
        row["id"]: row for row in _jsonl(observations) if row["split"] == "test" and row["id"] in trials
    }
    if set(originals) != set(trials):
        raise ValueError("Missing an original CompLex test rating")

    with np.load(predictions, allow_pickle=False) as archive:
        keys = archive["ids"].tolist()
        fitted = np.asarray(archive["predictions"], dtype=np.float64)
        if (
            len(set(keys)) != len(keys) or set(keys) != set(trials)
            or archive["methods"].tolist() != list(METHODS)
            or archive["fractions"].tolist() != list(FRACTIONS)
            or archive["seeds"].tolist() != list(SEEDS)
            or fitted.shape != (len(METHODS), len(FRACTIONS), len(SEEDS), len(keys))
            or not np.all((0 <= fitted) & (fitted <= 1))
        ):
            raise ValueError("Prediction archive differs from the fixed inventory")

    test_rows, y = [], []
    for key in keys:
        row, source = trials[key], originals[key]
        rating = source["observations"]["lexical_difficulty"]
        if (
            rating["status"] != "observed" or not 0 <= rating["value"] <= 1
            or row["target_text"] != source["target_span"]["text"]
            or any(row[k] != source[k] for k in ("stimulus_id", "kind", "corpus"))
        ):
            raise ValueError(f"Held-out metadata do not match for {key}")
        test_rows.append(row)
        y.append(rating["value"])

    expected = np.asarray(y)
    groups = np.array([row["context_group"] for row in test_rows])
    losses = np.abs(fitted - expected)
    kinds = np.array([row["kind"] for row in test_rows])
    seen = np.array([lexical_key(row) in training for row in test_rows])
    strata = {
        "single": kinds == "single", "multi": kinds == "multi", "target_seen_in_full_training": seen,
        "target_unseen_in_full_training": ~seen,
    }
    result = {
        "scope": "CompLex held-out sentence groups; average of five seed scores on the fixed test set.",
        "test_observations": len(expected),
        "test_sentence_groups": len(set(groups)),
        "input_sha256": {
            path.name: digest(path) for path in (design, observations, predictions)
        },
        "mae": _table(losses.mean(axis=-1)),
        "rmse": _table(np.sqrt(np.square(fitted - expected).mean(axis=-1))),
        "test_subgroups": {
            name: {"n": int(mask.sum()), "mae": _table(losses[..., mask].mean(axis=-1))}
            for name, mask in strata.items()
        },
        "condition_mae_differences": {
            f"{METHODS[b]}_minus_{METHODS[a]}": {
                str(fraction): _paired_delta(losses[b, i] - losses[a, i], groups)
                for i, fraction in enumerate(FRACTIONS)
            }
            for a, b in ((0, 1), (0, 2), (1, 2))
        },
        "fraction_mae_differences": {
            method: {
                f"{FRACTIONS[b]}_minus_{FRACTIONS[a]}": _paired_delta(
                    losses[i, b] - losses[i, a], groups,
                )
                for a, b in ((0, 1), (1, 2), (0, 2))
            }
            for i, method in enumerate(METHODS)
        },
        "same_target_context_ranking": _paired_context_ranking(test_rows, expected, fitted),
        "bootstrap": f"{BOOTSTRAP_DRAWS} paired held-out canonical sentence group draws, seed {BOOTSTRAP_SEED}.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("design", "observations", "predictions", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.design, args.observations, args.predictions, args.output)
