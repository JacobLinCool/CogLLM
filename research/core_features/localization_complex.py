"""Evaluate Stage 5 CompLex heads on held-out within-sentence localization."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from ..stage3.probe import _jsonl
from ..stage5.complex_inputs import digest
from ..stage5.complex_score import FRACTIONS, METHODS, SEEDS, TIE_TOLERANCE


BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 43


def prepare_rows(design, observations, predictions, historical_score):
    paths = (design, observations, predictions)
    hashes = {path.name: digest(path) for path in paths}
    saved = json.loads(historical_score.read_text(encoding="utf-8"))
    if hashes != saved["input_sha256"]:
        raise ValueError("Inputs differ from the archive used for Stage 5 scoring")

    trials = {row["id"]: row for row in _jsonl(design) if row["split"] == "test"}
    originals = {row["id"]: row for row in _jsonl(observations) if row["id"] in trials}
    with np.load(predictions, allow_pickle=False) as archive:
        keys = archive["ids"].tolist()
        estimates = archive["predictions"]
        if (
            len(keys) != saved["test_observations"] or len(keys) != len(set(keys))
            or set(keys) != set(trials) or set(keys) != set(originals)
            or archive["methods"].tolist() != list(METHODS)
            or archive["fractions"].tolist() != list(FRACTIONS)
            or archive["seeds"].tolist() != list(SEEDS)
            or estimates.shape != (len(METHODS), len(FRACTIONS), len(SEEDS), len(keys))
            or not np.all((0 <= estimates) & (estimates <= 1))
        ):
            raise ValueError("Prediction inventory differs from the verified Stage 5 predictions")

    # All seeds use the same fit when the entire training split is used.
    full_training = estimates[:, FRACTIONS.index(100)]
    if not np.all(full_training == full_training[:, :1]):
        raise ValueError("Expected the single full-data fit to be shared across seeds")

    grouped = defaultdict(list)
    expected = []
    spans = []
    for i, key in enumerate(keys):
        trial, original = trials[key], originals[key]
        rating = original["observations"]["lexical_difficulty"]
        span = original["target_span"]
        start, end = span["start"], span["end"]
        if (
            trial["stimulus_id"] != original["stimulus_id"] or trial["value"] is not None
            or trial["target_text"] != span["text"] or rating["status"] != "observed"
            or not 0 <= rating["value"] <= 1 or not 0 <= start < end
            or original["split"] != "test" or not original["exposure"]["target_designated"]
        ):
            raise ValueError(f"Unexpected target annotation: {key}")
        grouped[trial["stimulus_id"]].append(i)
        expected.append(rating["value"])
        spans.append((start, end))
    return grouped, np.array(expected), spans, full_training[:, 0], hashes


def score_pairs(grouped, expected, spans, estimates):
    left, right, clusters = [], [], []
    overlaps = equal_ratings = 0
    for sentence, candidates in grouped.items():
        for i, j in combinations(candidates, 2):
            start_i, end_i = spans[i]
            start_j, end_j = spans[j]
            if max(start_i, start_j) < min(end_i, end_j):
                overlaps += 1
            elif abs(expected[i] - expected[j]) <= TIE_TOLERANCE:
                equal_ratings += 1
            else:
                left.append(i)
                right.append(j)
                clusters.append(sentence)
    if not clusters:
        raise ValueError("No comparable held-out spans in the same sentence")

    direction = np.sign(expected[left] - expected[right])
    delta = estimates[:, left] - estimates[:, right]
    ties = np.abs(delta) <= TIE_TOLERANCE
    concordance = np.where(ties, 0.5, delta * direction > 0)
    _, groups, counts = np.unique(clusters, return_inverse=True, return_counts=True)
    sums = np.stack([np.bincount(groups, weights=method) for method in concordance])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    index = rng.integers(0, len(counts), size=(BOOTSTRAP_DRAWS, len(counts)))
    replicated = sums[:, index].sum(axis=-1) / counts[index].sum(axis=1)
    intervals = np.quantile(replicated, [0.025, 0.975], axis=1).T

    return {
        "held_out_rows": len(expected),
        "held_out_exact_sentences": len(grouped),
        "exact_sentences_with_at_least_two_rated_targets": sum(len(rows) > 1 for rows in grouped.values()),
        "exact_sentences_with_comparable_pairs": len(counts),
        "comparable_pairs": len(clusters),
        "excluded_overlapping_pairs": overlaps,
        "excluded_tied_human_ratings_in_disjoint_pairs": equal_ratings,
        "score": {
            method: {
                "concordance": float(concordance[i].mean()),
                "prediction_ties": int(ties[i].sum()),
                "sentence_bootstrap_95_interval": intervals[i].tolist(),
            }
            for i, method in enumerate(METHODS)
        },
        "paired_difference_vs_isolated_target": {
            method: {
                "concordance_difference": float((concordance[i] - concordance[0]).mean()),
                "sentence_bootstrap_95_interval": np.quantile(
                    replicated[i] - replicated[0], [0.025, 0.975],
                ).tolist(),
            }
            for i, method in enumerate(METHODS[1:], start=1)
        },
    }


def evaluate(design, observations, predictions, historical_score, output):
    grouped, expected, spans, estimates, hashes = prepare_rows(design, observations, predictions, historical_score)
    result = {
        "scope": "Ranking two distinct human-rated spans from an identical released CompLex test sentence.",
        "analysis": "Retrospective reanalysis of sealed Stage 5 L4 predictions. Unknown ratings for other spans are excluded.",
        "primary_contrast": "target_end_minus_target_only",
        "training_fraction": 100,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "exact sentence"},
        "score_definition": "Proportion of pairs ordered as the human means, with half credit for a prediction tie.",
        "input_sha256": {**hashes, "historical_score": digest(historical_score)},
        **score_pairs(grouped, expected, spans, estimates),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("design", "observations", "predictions", "historical-score", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    evaluate(**vars(parser.parse_args()))
