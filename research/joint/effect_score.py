"""Score matched joint weak-supervision predictions on the original holdouts."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .compare import squared_binary_losses
from .onestop_probe import row_losses


METHODS = ("linear_trial", "linear_effect", "shared_trial", "shared_effect")
COMPARISONS = {"linear": (0, 1), "shared": (2, 3)}
SEEDS = tuple(range(5))
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 43


def _summary(values):
    return {"mean": float(np.mean(values)), "by_seed": values.tolist()}


def _comparisons(values, draws):
    return {
        architecture: {
            "effect_minus_trial": _summary(values[with_effect] - values[direct]),
            "material_bootstrap_95_ci": np.quantile(
                (draws[with_effect] - draws[direct]).mean(axis=0), [0.025, 0.975],
            ).tolist(),
        }
        for architecture, (direct, with_effect) in COMPARISONS.items()
    }


def _resample(values, weights, groups, rng):
    """Draw whole material groups and form weighted means of held-out numerators."""
    eligible = weights > 0
    groups, group_by_row = np.unique(groups[eligible], return_inverse=True)
    cluster_counts = np.bincount(group_by_row, weights=weights[eligible])
    eligible_values = values[..., eligible]
    cluster_numerators = np.stack([
        eligible_values[..., group_by_row == i].sum(axis=-1) for i in range(len(groups))
    ], axis=-1)
    draws = rng.integers(0, len(groups), size=(BOOTSTRAP_DRAWS, len(groups)))
    return cluster_numerators[..., draws].sum(axis=-1) / cluster_counts[draws].sum(axis=-1)


def direct_metrics(losses, weights, groups):
    values = losses.sum(axis=-1) / weights.sum()
    draws = _resample(losses, weights, groups, np.random.default_rng(BOOTSTRAP_SEED))
    return {
        "by_method": {method: _summary(value) for method, value in zip(METHODS, values, strict=True)},
        "by_architecture": _comparisons(values, draws),
    }


def effect_metrics(human, models, groups, strata):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    human_draws = np.zeros(BOOTSTRAP_DRAWS)
    model_draws = np.zeros((*models.shape[:-1], BOOTSTRAP_DRAWS))
    human_mean = 0.0
    model_mean = np.zeros(models.shape[:-1])
    for coefficient, include in strata:
        expected = human[include]
        predicted = models[..., include]
        weights = np.ones(len(expected))
        clusters = groups[include]
        # Draw human and model outputs as one bundle to preserve paired materials.
        values = np.vstack((expected, predicted.reshape(-1, len(expected))))
        samples = _resample(values, weights, clusters, rng)
        human_draws += coefficient * samples[0]
        model_draws += coefficient * samples[1:].reshape(model_draws.shape)
        human_mean += coefficient * expected.mean()
        model_mean += coefficient * predicted.mean(axis=-1)
    abs_error = np.abs(model_mean - human_mean)
    return {
        "human": float(human_mean),
        "predicted_by_method": {key: _summary(row) for key, row in zip(METHODS, model_mean, strict=True)},
        "absolute_error_by_method": {key: _summary(row) for key, row in zip(METHODS, abs_error, strict=True)},
        "absolute_error_by_architecture": _comparisons(abs_error, np.abs(model_draws - human_draws)),
    }


def plaba_rewrite(ids, human, model, groups, source):
    lookup = {key: index for index, key in enumerate(ids)}
    positions = defaultdict(lambda: defaultdict(list))
    with source.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = (row["base_text_id"], int(row["mapped_orig_sentence_index"]))
            positions[key][row["variant"]].append(lookup[row["sentence_uid"]])

    truth, estimates, cluster_ids = [], [], []
    for item in positions.values():
        if not {"orig", "simp"} <= item.keys():
            continue
        original, simple = item["orig"], item["simp"]
        if len(set(groups[original + simple])) != 1:
            raise ValueError("A PLABA alignment crosses reader-connected material groups")
        truth.append(human[simple].mean() - human[original].mean())
        estimates.append(model[..., simple].mean(axis=-1) - model[..., original].mean(axis=-1))
        cluster_ids.append(groups[original[0]])
    if len(truth) != 265:
        raise ValueError("Expected the 265 Stage 2 PLABA rewrite alignments")
    return effect_metrics(np.array(truth), np.stack(estimates, axis=-1), np.array(cluster_ids), [(1, slice(None))])


def zhan_effect(source, predictions):
    families = source["zhan_families"]
    verbs = source["zhan_verb_type"]
    counts = source["zhan_counts"]
    if (
        not np.array_equal(families[::2], families[1::2])
        or not np.array_equal(verbs[::2], verbs[1::2])
        or len(set(families)) != 36
    ):
        raise ValueError("Expected aligned Zhan free/pronoun pairs")
    human = counts[:, 1] / counts.sum(axis=-1)
    return effect_metrics(
        human[1::2] - human[::2], predictions[..., 1::2] - predictions[..., ::2], families[::2],
        [(-1, verbs[::2] == "IC-1"), (1, verbs[::2] == "IC-2")],
    )


def evaluate(design, predictions, plaba_sentences, output):
    with np.load(design, allow_pickle=False) as source, np.load(predictions, allow_pickle=False) as data:
        if data["methods"].tolist() != list(METHODS) or data["seeds"].tolist() != list(SEEDS):
            raise ValueError("Prediction method and seed inventory does not match the fixed protocol")
        pred = {}
        for task, width in (("plaba", (4,)), ("zhan", ()), ("onestop", (4,))):
            ids = source[f"{task}_ids"]
            values = data[f"{task}_predictions"]
            if (
                not np.array_equal(ids, data[f"{task}_ids"])
                or values.shape != (len(METHODS), len(SEEDS), len(ids), *width)
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"Mismatch in {task} out-of-fold predictions")
            pred[task] = np.asarray(values, dtype=np.float64)
        ratings_pred, binary_pred = pred["plaba"][..., :2], pred["plaba"][..., 2:]
        if (
            not np.all((1 <= ratings_pred) & (ratings_pred <= 5))
            or not np.all((0 <= binary_pred) & (binary_pred <= 1))
            or not np.all((0 <= pred["zhan"]) & (pred["zhan"] <= 1))
            or not np.allclose(pred["onestop"].sum(axis=-1), 1, atol=1e-6, rtol=0)
        ):
            raise ValueError("Prediction scale or probability normalization is invalid")

        plaba = {}
        group = source["plaba_groups"]
        ratings = source["plaba_rating_counts"]
        observed_average = (ratings @ np.arange(1, 6)) / ratings.sum(axis=-1)
        for column, measure in enumerate(("processing_ease_sentence_mae", "understanding_sentence_mae")):
            plaba[measure] = direct_metrics(
                np.abs(pred["plaba"][..., column] - observed_average[:, column]), np.ones(len(group)), group,
            )
        for column, measure in enumerate(("any_issue_brier", "unclear_term_brier")):
            counts = source["plaba_issue_counts"][:, column]
            plaba[measure] = direct_metrics(
                squared_binary_losses(pred["plaba"][..., column + 2], counts), counts.sum(axis=-1), group,
            )
        counts = source["zhan_counts"]
        zhan = direct_metrics(
            squared_binary_losses(pred["zhan"], counts), counts.sum(axis=-1), source["zhan_families"],
        )
        counts = source["onestop_counts"]
        correct = source["onestop_correct"]
        totals = counts.sum(axis=-1)
        losses = [[row_losses(seed, counts, correct) for seed in arm] for arm in pred["onestop"]]
        onestop = {
            measure: direct_metrics(
                np.array([[seed[measure] for seed in arm] for arm in losses]), totals, source["onestop_families"],
            )
            for measure in losses[0][0]
        }
        effects = {
            "plaba_rating_points_simplified_minus_original": plaba_rewrite(
                source["plaba_ids"], observed_average[:, 1], pred["plaba"][..., 1], group, plaba_sentences,
            ),
            "zhan_interaction_ic2_minus_ic1": zhan_effect(source, pred["zhan"]),
        }
        linear_difference = float(np.abs(pred["onestop"][1] - pred["onestop"][0]).max())
        if linear_difference > 1e-7:
            raise ValueError(f"OneStop linear predictions changed without any OneStop effect: {linear_difference}")

    result = {
        "scope": "Within-study held-out materials; fixed reader panels, folds and five seed trials.",
        "bootstrap": f"Draw {BOOTSTRAP_DRAWS} paired material groups, seed {BOOTSTRAP_SEED}, conditional on saved fits.",
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("design", design), ("predictions", predictions), ("plaba_sentences", plaba_sentences))
        },
        "effects": effects,
        "direct": {"plaba": plaba, "zhan_choice_brier": zhan, "onestop": onestop},
        "max_linear_onestop_probability_difference": linear_difference,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("design", "predictions", "plaba-sentences", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.design, args.predictions, args.plaba_sentences, args.output)
