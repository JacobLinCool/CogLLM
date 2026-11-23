"""Audit the joint-model holdout predictions on each task's own units."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .onestop_probe import row_losses


METHODS = ("linear", "independent_adapter", "shared_adapter")
DRAWS = 10000
SEED = 43


def paired_errors(losses, denominator, groups):
    active = denominator > 0
    _, index = np.unique(groups[active], return_inverse=True)
    cluster_denominator = np.bincount(index, weights=denominator[active])
    delta = np.bincount(index, weights=(losses[2, active] - losses[1, active]))
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(delta), size=(DRAWS, len(delta)))
    ratios = delta[sampled].sum(axis=-1) / cluster_denominator[sampled].sum(axis=-1)
    return {
        "by_method": dict(zip(METHODS, (losses.sum(axis=-1) / denominator.sum()).tolist(), strict=True)),
        "shared_minus_independent": float(delta.sum() / denominator.sum()),
        "material_bootstrap_95_ci": np.quantile(ratios, [0.025, 0.975]).tolist(),
        "sampled_material_groups": len(cluster_denominator),
    }


def squared_binary_losses(p, counts):
    return counts[:, 0] * p**2 + counts[:, 1] * (1 - p)**2


def paired_rewrite(ids, human, model, source):
    lookup = {key: index for index, key in enumerate(ids)}
    anchors = defaultdict(lambda: defaultdict(list))
    with source.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            sentence = lookup[row["sentence_uid"]]
            anchors[(row["base_text_id"], row["mapped_orig_sentence_index"])][row["variant"]].append(sentence)
    human_and_model = np.vstack((human, model))
    shifts = [
        human_and_model[:, item["simp"]].mean(axis=-1) - human_and_model[:, item["orig"]].mean(axis=-1)
        for item in anchors.values() if {"orig", "simp"} <= item.keys()
    ]
    if len(shifts) != 265:
        raise ValueError("PLABA rewrite alignment differs from the verified Stage 2 contrast")
    means = np.array(shifts).mean(axis=0)
    return {
        "aligned_positions": len(shifts),
        "mean_simplified_minus_original_understanding_rating": dict(
            zip(("human", *METHODS), means.tolist(), strict=True)
        ),
    }


def zhan_interaction(probabilities, counts, ids, verbs):
    if any(
        free.removesuffix("-free") != pronoun.removesuffix("-pronoun")
        for free, pronoun in zip(ids[::2], ids[1::2], strict=True)
    ) or not np.array_equal(verbs[::2], verbs[1::2]):
        raise ValueError("Zhan rows must be ordered in aligned free/pronoun pairs")
    human = counts[:, 1] / counts.sum(axis=-1)
    values = np.vstack((human, probabilities))
    deltas = values[:, 1::2] - values[:, ::2]
    grouped = [deltas[:, verbs[::2] == group].mean(axis=-1) for group in ("IC-1", "IC-2")]
    return {
        "interaction_ic2_minus_ic1": dict(zip(("human", *METHODS), (grouped[1] - grouped[0]).tolist(), strict=True)),
    }


def evaluate(design, predictions, plaba_sentences, output):
    with np.load(design, allow_pickle=False) as source, np.load(predictions, allow_pickle=False) as fitted:
        if fitted["methods"].tolist() != list(METHODS):
            raise ValueError("Expected linear, independent, and shared readout predictions")
        p = {}
        for task, expected in (("plaba", (3, 609, 4)), ("zhan", (3, 72)), ("onestop", (3, 1872, 4))):
            p[task] = fitted[f"{task}_predictions"]
            if p[task].shape != expected or not np.array_equal(fitted[f"{task}_ids"], source[f"{task}_ids"]):
                raise ValueError(f"Unexpected predictions for {task}")
        if (
            not np.isfinite(p["plaba"]).all()
            or not ((1 <= p["plaba"][..., :2]) & (p["plaba"][..., :2] <= 5)).all()
            or not ((0 <= p["plaba"][..., 2:]) & (p["plaba"][..., 2:] <= 1)).all()
            or not ((0 <= p["zhan"]) & (p["zhan"] <= 1)).all()
            or not np.allclose(p["onestop"].sum(axis=-1), 1, atol=1e-6, rtol=0)
        ):
            raise ValueError("Invalid output scale or probability")

        ratings = source["plaba_rating_counts"]
        average = (ratings @ np.arange(1, 6)) / ratings.sum(axis=-1)
        groups = source["plaba_groups"]
        plaba = {}
        for column, name in enumerate(("processing_ease_average_sentence_mae", "understanding_average_sentence_mae")):
            plaba[name] = paired_errors(np.abs(p["plaba"][..., column] - average[:, column]), np.ones(len(groups)), groups)
        for column, name in enumerate(("any_issue_brier", "unclear_term_brier")):
            counts = source["plaba_issue_counts"][:, column]
            issue = p["plaba"][..., column + 2]
            plaba[name] = paired_errors(squared_binary_losses(issue, counts), counts.sum(axis=-1), groups)

        counts = source["zhan_counts"]
        zhan = {
            "choice_brier": paired_errors(squared_binary_losses(p["zhan"], counts), counts.sum(axis=-1), source["zhan_families"]),
            **zhan_interaction(p["zhan"], counts, source["zhan_ids"], source["zhan_verb_type"]),
        }
        counts = source["onestop_counts"]
        correct = source["onestop_correct"]
        selected_losses = [row_losses(model, counts, correct) for model in p["onestop"]]
        onestop = {
            key: paired_errors(
                np.stack([loss[key] for loss in selected_losses]), counts.sum(axis=-1), source["onestop_families"],
            )
            for key in selected_losses[0]
        }
        rewrite = paired_rewrite(source["plaba_ids"], average[:, 1], p["plaba"][..., 1], plaba_sentences)

    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("design", design), ("predictions", predictions), ("plaba_sentences", plaba_sentences))
        },
        "bootstrap": f"{DRAWS} paired draws of original material groups on fixed fits, reader panel, and seed.",
        "plaba": {**plaba, "rewrite_contrast": rewrite},
        "zhan": zhan,
        "onestop": onestop,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("design", "predictions", "plaba-sentences", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.design, args.predictions, args.plaba_sentences, args.output)
