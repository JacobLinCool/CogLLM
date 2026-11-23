"""Score a shared pronoun-choice readout at the two Zhan Experiment 1 stopping points."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .issue_probe import _fit_logistic, _ridge_design
from .probe import _jsonl


CONDITIONS = ("free", "pronoun")
VERB_TYPES = ("IC-1", "IC-2")
REFERENCE = "before_pronoun_plus_prompt_type"
LANDMARK = "observed_stop_hidden"
LANDMARK_WITH_FLAG = "observed_stop_hidden_plus_prompt_type"


def load_trials(data):
    stimuli = defaultdict(dict)
    for row in _jsonl(data / "stimuli.jsonl"):
        family = stimuli[row["family_id"]]
        prompt_type = row["metadata"]["prompt_type"]
        if prompt_type not in CONDITIONS or prompt_type in family:
            raise ValueError(f"Invalid prompt condition for {row['id']}")
        family[prompt_type] = row

    family_ids = sorted(stimuli)
    pairs = [[stimuli[family][condition] for condition in CONDITIONS] for family in family_ids]
    by_id = {row["id"]: row for pair in pairs for row in pair}
    ids = list(by_id)
    counts = {stimulus_id: [0, 0] for stimulus_id in ids}
    keys = set()
    for row in _jsonl(data / "trials.jsonl"):
        stimulus = by_id[row["stimulus_id"]]
        key = (row["participant_id"], stimulus["family_id"])
        choice = row["observations"]["reMention"]
        if (
            key in keys
            or row["exposure"]["task"] != "explanation_continuation"
            or row["exposure"]["visible_end"] != len(stimulus["text"])
            or row["task_context"]["prompt_type"] != stimulus["metadata"]["prompt_type"]
            or choice["status"] != "observed"
            or choice["value"] not in ("np1", "np2")
        ):
            raise ValueError(f"Invalid eligible response for {row['id']}")
        keys.add(key)
        count = counts[row["stimulus_id"]]
        count[0] += choice["value"] == "np1"
        count[1] += 1
    counts = np.array([counts[stimulus_id] for stimulus_id in ids]).reshape(-1, 2, 2)
    if counts[:, :, 1].sum() != 1651 or len({reader for reader, _ in keys}) != 50 or (counts[:, :, 1] == 0).any():
        raise ValueError("Unexpected published Zhan Experiment 1 denominators")

    verb_types = np.array([free["metadata"]["verb_type"] for free, _ in pairs])
    if set(verb_types) != set(VERB_TYPES) or any(
        pronoun["metadata"]["verb_type"] != verb for (_, pronoun), verb in zip(pairs, verb_types, strict=True)
    ):
        raise ValueError("Unexpected or inconsistent verb type")
    splits = json.loads((data / "splits.json").read_text(encoding="utf-8"))["folds"]
    held_out = [(family, fold["fold"]) for fold in splits for family in fold["families"]]
    group_folds = dict(held_out)
    if (
        set(group_folds) != set(family_ids) or len(held_out) != len(family_ids)
        or len(family_ids) != 36 or set(group_folds.values()) != set(range(5))
    ):
        raise ValueError("Expected the locked five folds of 36 stimulus families")
    return family_ids, ids, counts, verb_types, np.array([group_folds[family] for family in family_ids])


def predict(features, ids, positives, totals, verb_types, folds):
    with np.load(features, allow_pickle=False) as archive:
        hidden = np.asarray(archive["hidden"], dtype=np.float64)
        cached_ids = archive["ids"].tolist()
    indices = {stimulus_id: index for index, stimulus_id in enumerate(cached_ids)}
    if len(indices) != len(cached_ids) or set(indices) != set(ids) or hidden.shape != (len(ids), 2048):
        raise ValueError("Expected one 2,048-dimensional observation per stimulus")
    hidden = hidden[[indices[stimulus_id] for stimulus_id in ids]]
    norm = np.linalg.norm(hidden, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError("Invalid frozen hidden state")
    hidden /= norm
    condition = np.tile([0.0, 1.0], len(folds))
    verb = np.repeat(verb_types == "IC-2", 2).astype(np.float64)
    designs = {
        "prompt_type": condition[:, None],
        REFERENCE: np.column_stack((np.repeat(hidden[::2], 2, axis=0), condition)),
        LANDMARK: hidden,
        LANDMARK_WITH_FLAG: np.column_stack((hidden, condition)),
        "author_verb_type_plus_prompt_interaction": np.column_stack((condition, verb, condition * verb)),
    }
    folds = np.repeat(folds, 2)
    predictions = {}
    for method, design in designs.items():
        scores = np.empty(len(design))
        for fold in range(5):
            test = folds == fold
            train = ~test
            train_design, test_design = _ridge_design(design[train], design[test])
            scores[test] = _fit_logistic(train_design, test_design, positives.ravel()[train], totals.ravel()[train])
        predictions[method] = scores.reshape(totals.shape)
    return predictions


def evaluate(data, features, output):
    families, ids, counts, verbs, folds = load_trials(data)
    positives, totals = counts[..., 0], counts[..., 1]
    estimates = predict(features, ids, positives, totals, verbs, folds)
    rates = positives / totals
    human_shift = rates[:, 1] - rates[:, 0]
    condition_losses = {
        method: positives * (1 - scores)**2 + (totals - positives) * scores**2
        for method, scores in estimates.items()
    }
    losses = {method: values.sum(axis=1) for method, values in condition_losses.items()}
    group_trials = totals.sum(axis=1)
    rng = np.random.default_rng(43)
    by_verb = {verb: np.flatnonzero(verbs == verb) for verb in VERB_TYPES}
    draws = {
        verb: rng.choice(indices, size=(10000, len(indices)), replace=True)
        for verb, indices in by_verb.items()
    }
    overall_draws = np.column_stack(tuple(draws.values()))
    contrasts = {}
    for method in (LANDMARK, LANDMARK_WITH_FLAG):
        loss_delta = losses[method] - losses[REFERENCE]
        contrasts[method] = {
            "difference": float(loss_delta.sum() / group_trials.sum()),
            "95_ci": np.quantile(
                loss_delta[overall_draws].sum(axis=1) / group_trials[overall_draws].sum(axis=1), [0.025, 0.975],
            ).tolist(),
        }

    methods = {}
    for method, scores in estimates.items():
        shift = scores[:, 1] - scores[:, 0]
        methods[method] = {
            "brier": float(losses[method].sum() / group_trials.sum()),
            "condition_brier": dict(zip(
                CONDITIONS, (condition_losses[method].sum(axis=0) / totals.sum(axis=0)).tolist(), strict=True,
            )),
            "mean_absolute_item_shift_error": float(np.abs(shift - human_shift).mean()),
            "shift_by_verb_type": {
                verb: {
                    "mean": float(shift[indices].mean()),
                    "error_vs_human": float((shift - human_shift)[indices].mean()),
                    "error_vs_human_95_ci": np.quantile((shift - human_shift)[draws[verb]].mean(axis=1), [0.025, 0.975]).tolist(),
                }
                for verb, indices in by_verb.items()
            },
        }
    inputs = {name: data / name for name in ("stimuli.jsonl", "trials.jsonl", "splits.json")}
    inputs["features.npz"] = features
    metrics = {
        "experiment": "Zhan et al. (2020), Experiment 1, eligible continuation choices",
        "observed_trials": int(totals.sum()),
        "stimulus_families": len(families),
        "folds": 5,
        "reader_scope": "Overlapping readers; held-out stimulus families.",
        "ridge_penalty": 1.0,
        "bootstrap": "10,000 paired family draws stratified by verb; fixed reader panel and fitted out-of-fold predictions.",
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()},
        "human_by_verb_type": {
            verb: {
                "families": len(indices),
                "total_trials": totals[indices].sum(axis=0).tolist(),
                "subject_choices": positives[indices].sum(axis=0).tolist(),
                "item_mean_shift": float(human_shift[indices].mean()),
                "item_mean_shift_95_ci": np.quantile(human_shift[draws[verb]].mean(axis=1), [0.025, 0.975]).tolist(),
            }
            for verb, indices in by_verb.items()
        },
        "methods": methods,
        "brier_differences_vs_before_pronoun_plus_prompt_type": contrasts,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, family in enumerate(families):
            row = {
                "family_id": family,
                "verb_type": str(verbs[index]),
                "fold": int(folds[index]),
                "conditions": list(CONDITIONS),
                "subject_choices": positives[index].tolist(),
                "trials": totals[index].tolist(),
                "predicted_subject_rates": {method: scores[index].tolist() for method, scores in estimates.items()},
            }
            handle.write(json.dumps(row, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.data, args.features, args.output)
