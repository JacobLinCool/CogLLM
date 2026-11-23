"""Score sealed first-visit and reread-destination forecasts on BSPR Experiment 1."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..stage3.probe import _jsonl


EVENT_METHODS = ("event_prior", "event_position", "event_qwen")
SOURCE_METHODS = ("source_distance", "source_qwen")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 43


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def position(row):
    return row["id"], row["stop"]


def load_forecasts(path):
    expected = {"id", "stop", "source_stops", *EVENT_METHODS, *SOURCE_METHODS}
    predictions = {}
    for row in _jsonl(path):
        if row.keys() != expected:
            raise ValueError("Unexpected BSPR forecast fields")
        key = position(row)
        stops = row["source_stops"]
        if (
            key in predictions or type(row["stop"]) is not int or not stops
            or stops != sorted(set(stops))
            or any(type(stop) is not int or not 0 < stop < row["stop"] for stop in stops)
        ):
            raise ValueError(f"Duplicate anchor or invalid destination offsets for {key}")
        if any(not 0 < row[method] < 1 for method in EVENT_METHODS):
            raise ValueError(f"Readback probability outside (0, 1) for {key}")
        for method in SOURCE_METHODS:
            p = np.asarray(row[method], dtype=np.float64)
            if p.shape != (len(stops),) or not np.all(np.isfinite(p) & (p > 0) & (p <= 1)) or not math.isclose(
                float(p.sum()), 1.0, abs_tol=1e-7, rel_tol=0.0,
            ):
                raise ValueError(f"Invalid {method} destination distribution for {key}")
            row[method] = p
        predictions[key] = row
    return predictions


def aligned_holdout(path, predictions, excluded_stimuli):
    rows = list(_jsonl(path))
    train = {row["group_id"] for row in rows if row["split"] == "train"}
    holdout = [row for row in rows if row["split"] == "test"]
    holdout_groups = {row["group_id"] for row in holdout}
    if len(rows) != len({(row["trial_id"], row["stop"]) for row in rows}) or train & holdout_groups:
        raise ValueError("Expected unique first visits and disjoint train/test material families")
    if len(holdout_groups) != 5 or len(train) != 15:
        raise ValueError("Expected the fixed 15 training/5 held-out BSPR material families")
    eligible = [row for row in holdout if row["id"] not in excluded_stimuli]
    if {position(row) for row in eligible} != predictions.keys():
        raise ValueError("Prediction coverage differs from complete certified held-out anchors")
    for row in eligible:
        forecast = predictions[position(row)]
        if type(row["readback"]) is not bool or len(forecast["source_stops"]) != row["region"] - 1:
            raise ValueError(f"Forecast omits observed earlier regions for {position(row)}")
        if row["readback"] and row["source_stop"] not in forecast["source_stops"]:
            raise ValueError(f"Destination missing from its choice set for {position(row)}")
        if not row["readback"] and row["source_stop"] is not None:
            raise ValueError("A forward transition cannot carry a reread destination")
    return holdout, eligible


def average_precision(probability, observed):
    order = np.argsort(-probability, kind="stable")
    ordered = probability[order]
    # All trials at the same threshold enter together, including an intercept-only model.
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(order) - 1]
    true_positive = np.cumsum(observed[order])[ends]
    recall = true_positive / true_positive[-1]
    precision = true_positive / (ends + 1)
    return float(np.dot(np.diff(recall, prepend=0), precision))


def paired_material_interval(values, rows):
    groups, index = np.unique([row["group_id"] for row in rows], return_inverse=True)
    if len(groups) < 2:
        raise ValueError("A paired material interval requires at least two held-out families")
    counts = np.bincount(index)
    effects = np.bincount(index, weights=values)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    clusters = rng.integers(len(groups), size=(BOOTSTRAP_DRAWS, len(groups)))
    replicated = effects[clusters].sum(axis=1) / counts[clusters].sum(axis=1)
    return {
        "difference": float(np.mean(values)),
        "descriptive_paired_material_95_ci": np.quantile(replicated, [0.025, 0.975]).tolist(),
        "groups": len(groups),
    }


def score_events(rows, predictions):
    observed = np.array([row["readback"] for row in rows])
    if not 0 < observed.sum() < len(rows):
        raise ValueError("Both observed first-visit outcomes are needed")
    metrics = {}
    losses = {}
    for method in EVENT_METHODS:
        p = np.array([predictions[position(row)][method] for row in rows])
        brier = (p - observed) ** 2
        nll = -np.log(np.where(observed, p, 1 - p))
        losses[method] = brier
        metrics[method] = {
            "brier": float(brier.mean()),
            "nll_nats": float(nll.mean()),
            "average_precision": average_precision(p, observed),
        }
    contrast = paired_material_interval(losses["event_qwen"] - losses["event_position"], rows)
    return {"metrics": metrics, "brier_qwen_minus_position": contrast}


def score_destinations(rows, predictions):
    nlls = {method: [] for method in SOURCE_METHODS}
    ranked = {method: [] for method in (*SOURCE_METHODS, "previous_region")}
    region_errors = {method: [] for method in ranked}
    for row in rows:
        forecast = predictions[position(row)]
        stops = forecast["source_stops"]
        target = stops.index(row["source_stop"])
        previous = len(stops) - 1
        ranked["previous_region"].append(float(target == previous))
        region_errors["previous_region"].append(float(previous - target))
        for method in SOURCE_METHODS:
            p = forecast[method]
            top = np.isclose(p, p.max(), atol=1e-12, rtol=0)
            nlls[method].append(-math.log(p[target]))
            ranked[method].append(float(top[target] / top.sum()))
            expected = np.dot(np.arange(len(stops)), p)
            region_errors[method].append(abs(float(expected) - target))
    metrics = {
        method: {
            "top_1_fractional_ties": float(np.mean(ranked[method])),
            "expected_destination_mae_regions": float(np.mean(region_errors[method])),
        }
        for method in ranked
    }
    for method, losses in nlls.items():
        metrics[method]["nll_nats"] = float(np.mean(losses))
    contrast = paired_material_interval(np.subtract(nlls["source_qwen"], nlls["source_distance"]), rows)
    return {"metrics": metrics, "nll_qwen_minus_distance": contrast}


def evaluate(outcomes_path, predictions_path, seal_path, extraction_path, output):
    prediction_sha = digest(predictions_path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if prediction_sha != seal["predictions_sha256"]:
        raise ValueError("Forecast file does not match the pre-scoring seal")
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    excluded_stimuli = {stop["id"] for stop in extraction["excluded_stops"]}
    predictions = load_forecasts(predictions_path)
    intended, eligible = aligned_holdout(outcomes_path, predictions, excluded_stimuli)
    positive = [row for row in eligible if row["readback"]]
    positive_with_multiple_sources = [row for row in positive if row["region"] > 2]
    result = {
        "data": "BSPR Experiment 1 discourse stimuli",
        "scope": "First-visit movement within subsequently alignable displays, with five held-out material families.",
        "first_visit_target": "Observed backward keypress or restart versus observed advance by one region.",
        "destination_target": "Furthest region reached in the contiguous backward movement, conditional on readback.",
        "coverage": {
            "holdout_material_families": sorted({row["group_id"] for row in intended}),
            "scored_material_families": sorted({row["group_id"] for row in eligible}),
            "scored_readers": len({row["reader_id"] for row in eligible}),
            "intended_first_visits": len(intended),
            "scored_first_visits": len(eligible),
            "scored_backward_episodes": len(positive),
            "scored_unique_stimulus_stops": len(predictions),
            "scored_backward_episodes_with_two_or_more_earlier_regions": len(positive_with_multiple_sources),
            "positive_followed_by_progress_beyond_origin": sum(row["resumed_beyond_stop"] for row in positive),
            "rejected_holdout_stimuli": sorted({row["id"] for row in intended} & excluded_stimuli),
        },
        "events": score_events(eligible, predictions),
        "destinations": score_destinations(positive, predictions),
        "destinations_with_multiple_sources": score_destinations(positive_with_multiple_sources, predictions),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
            "resampled": "paired item families; predictions and readers held fixed",
        },
        "input_sha256": {
            "outcomes": digest(outcomes_path), "forecast": prediction_sha,
            "seal": digest(seal_path), "extraction": digest(extraction_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("outcomes", "predictions", "seal", "extraction", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.outcomes, args.predictions, args.seal, args.extraction, args.output)
