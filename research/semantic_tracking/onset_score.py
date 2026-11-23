"""Evaluate the first deliberate backward action at exact German display stops."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.core_features.readback_score import (
    EVENT_METHODS, digest, load_forecasts, position, score_destinations, score_events,
)
from research.stage3.probe import _jsonl


def one_word(doc, region):
    start = doc["stops"][region - 2] + 1 if region > 1 else 0
    return len(doc["text"][start:doc["stops"][region - 1]].split()) == 1


def region_text(doc, region):
    start = doc["stops"][region - 2] + 1 if region > 1 else 0
    return doc["text"][start:doc["stops"][region - 1]]


def interval(values):
    if len(values) < 2:
        raise ValueError("Expected at least two matched test families")
    rng = np.random.default_rng(43)
    indices = rng.integers(len(values), size=(10_000, len(values)))
    replicates = values[indices].mean(axis=1)
    return np.quantile(replicates, (0.025, 0.975)).tolist()


def motion_scores(rows, predictions):
    return {
        "at_risk_visits": len(rows),
        "first_backward_actions": sum(row["readback"] for row in rows),
        "item_families": len({row["group_id"] for row in rows}),
        **score_events(rows, predictions),
    }


def destination_scores(rows, predictions):
    return {
        "first_backward_actions": len(rows),
        "item_families": len({row["group_id"] for row in rows}),
        **score_destinations(rows, predictions),
    }


def matched_effect(rows, predictions, stimuli, offset, match_wording=True):
    by_family = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["critical_offset"] == offset:
            by_family[row["group_id"]][row["condition"]].append(row)
    per_family = []
    for family in sorted(by_family):
        arms = by_family[family]
        if set(arms) != {"early", "late"}:
            continue
        snippets = {
            region_text(stimuli[row["id"]], row["region"])
            for arm in arms.values() for row in arm
        }
        if match_wording and len(snippets) != 1:
            continue
        terms = ["observed", *EVENT_METHODS]
        delta = {}
        for method in terms:
            arm_means = {
                arm: np.mean([
                    float(row["readback"]) if method == "observed" else predictions[position(row)][method]
                    for row in group
                ])
                for arm, group in arms.items()
            }
            delta[method] = arm_means["late"] - arm_means["early"]
        per_family.append((family, delta, sum(map(len, arms.values()))))
    if len(per_family) < 2:
        return {"available": False, "matched_item_families": len(per_family)}
    terms = ["observed", *EVENT_METHODS]
    point_and_interval = {}
    for method in terms:
        difference = np.asarray([values[method] for _, values, _ in per_family])
        point_and_interval[method] = {
            "late_minus_early_pp": float(100 * difference.mean()),
            "paired_item_95_interval_pp": [100 * x for x in interval(difference)],
        }
    return {
        "available": True, "matched_item_families": len(per_family),
        "at_risk_visits": sum(visits for _, _, visits in per_family),
        "exactly_same_region_text_required": match_wording,
        "condition_contrast": point_and_interval,
    }


def run(outcomes, stimuli, forecasts, seal, extraction, output):
    forecast_id = digest(forecasts)
    if json.loads(seal.read_text(encoding="utf-8"))["predictions_sha256"] != forecast_id:
        raise ValueError("Forecasts do not match their pre-scoring seal")
    inventory = list(_jsonl(stimuli))
    docs = {row["id"]: row for row in inventory}
    if len(docs) != len(inventory):
        raise ValueError("Repeated stimulus IDs")
    exclusions = {
        row["id"] for row in json.loads(extraction.read_text(encoding="utf-8"))["excluded_stops"]
    } & docs.keys()
    released = list(_jsonl(outcomes))
    if any(row["split"] != "test" or not row["initial_risk"] for row in released):
        raise ValueError("Expected only held-out first observed backward action risk rows")
    if len({(row["trial_id"], row["region"]) for row in released}) != len(released):
        raise ValueError("Repeated held-out trial/position observation")
    rows = [row for row in released if row["id"] not in exclusions]
    predictions = load_forecasts(forecasts)
    if predictions.keys() != {position(row) for row in rows}:
        raise ValueError("A forecast is missing for a valid evaluation anchor or an extra anchor is present")
    positives = [row for row in rows if row["readback"]]
    if len({row["trial_id"] for row in positives}) != len(positives):
        raise ValueError("The response must be the first backward action in its trial")
    for row in rows:
        doc = docs[row["id"]]
        index = row["region"] - 1
        prediction = predictions[position(row)]
        if (doc["stops"][index] != row["stop"] or prediction["source_stops"] != doc["stops"][:index]
                or (row["region_words"] == 1) != one_word(doc, row["region"])
                or row["readback"] and row["source_stop"] not in prediction["source_stops"]):
            raise ValueError(f"Misaligned observed coordinates for {position(row)}")
    exact_start = [row for row in rows if row["region_words"] == 1]
    exact_critical = [row for row in exact_start if row["critical"]]
    multiple = [row for row in positives if row["region"] > 2]
    exact_both = [
        row for row in multiple
        if row["region_words"] == 1
        and one_word(docs[row["id"]], docs[row["id"]]["stops"].index(row["source_stop"]) + 1)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "source": "Paape and Vasishth Experiment 2 German bidirectional self-paced reading",
        "scope": "Explicit first backward keypress or restart, conditional on reaching an eligible displayed region with no preceding backward action. Earlier destination is behavior and was not reported as a semantic antecedent.",
        "display_words": "Single-whitespace-delimited word in the experiment's visible region. Anchors use the certified causal ending token.",
        "coverage": {
            "planned_first_backward_risk_set_visits": len(released),
            "heldout_items": sorted({row["group_id"] for row in rows}),
            "scored_unique_anchors": len(predictions), "scored_readers": len({row["reader_id"] for row in rows}),
            "excluded_stimuli": sorted(exclusions),
            "first_backward_then_progressed_beyond_origin": sum(row["resumed_beyond_stop"] for row in positives),
        },
        "event": {
            "all": motion_scores(rows, predictions),
            "single_word": motion_scores(exact_start, predictions),
            "single_word_critical_verb": motion_scores(exact_critical, predictions),
        },
        "destination": {
            "all": destination_scores(positives, predictions),
            "choice_set_size_at_least_two": destination_scores(multiple, predictions),
            "single_word_origin_and_destination_choice_set_size_at_least_two": destination_scores(exact_both, predictions),
        },
        "manipulation_matched_wording": {
            str(offset): matched_effect(rows, predictions, docs, offset)
            for offset in (-1, 0, 1)
        },
        "manipulation_note": "Each equal-weight contrast compares late minus early disambiguation in material pairs with an identically worded region. Both experiment arms must contribute at-risk observations. The preceding region is a diagnostic control for localization and can still differ in its preceding context.",
        "input_sha256": {
            "outcomes": digest(outcomes), "stimuli": digest(stimuli), "forecasts": forecast_id,
            "seal": digest(seal), "extraction": digest(extraction),
        },
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("outcomes", "stimuli", "forecasts", "seal", "extraction", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    run(**vars(parser.parse_args()))
