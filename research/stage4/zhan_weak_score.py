"""Evaluate held-out Zhan Experiment 1 continuation predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..stage3.zhan_landmark_probe import CONDITIONS, VERB_TYPES, load_trials
from .zhan_weak import ARMS, SEEDS


BOOTSTRAP_SEED = 43
BOOTSTRAP_DRAWS = 10_000


def _load_predictions(path, family_ids):
    with np.load(path, allow_pickle=False) as data:
        if (
            data["family_ids"].tolist() != family_ids
            or data["arms"].tolist() != list(ARMS)
            or data["seeds"].tolist() != list(SEEDS)
        ):
            raise ValueError("Expected the ordered Zhan stimulus families, fixed evaluation arms and label seeds")
        probabilities = data["predictions"]
    expected_shape = (len(ARMS), len(SEEDS), len(family_ids), len(CONDITIONS))
    if probabilities.shape != expected_shape or probabilities.dtype != np.float64:
        raise ValueError(f"Expected float64 probabilities with shape {expected_shape}")
    if not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("Each held-out probability must be finite and in [0, 1]")
    if not np.all(probabilities[2] == probabilities[2, :1]):
        raise ValueError("The full-label predictions must be identical across label draws")
    return probabilities


def _summarize(values):
    return {
        "mean": float(values.mean()),
        "range": [float(values.min()), float(values.max())],
        "by_label_draw": values.tolist(),
    }


def _contrast(per_label_draw, bootstrap_values):
    return {
        "difference": _summarize(per_label_draw),
        "material_bootstrap_95_ci": np.quantile(bootstrap_values, [0.025, 0.975]).tolist(),
    }


def _paired_intervals(brier_difference, total, item_shift_errors, strata):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [rng.choice(group, size=(BOOTSTRAP_DRAWS, len(group)), replace=True) for group in strata]

    all_draws = np.column_stack(draws)
    brier_samples = brier_difference.mean(axis=0)[all_draws].sum(axis=1) / total[all_draws].sum(axis=1)

    # Rebuild the two group means before taking the absolute interaction error
    # in each matched resample and label draw.
    resampled_group_errors = [item_shift_errors[:, :, draw].mean(axis=-1) for draw in draws]
    resampled_abs_errors = np.abs(resampled_group_errors[1] - resampled_group_errors[0])
    interaction_samples = (resampled_abs_errors[1] - resampled_abs_errors[0]).mean(axis=0)
    return brier_samples, interaction_samples


def evaluate(data, heads, output):
    families, _, counts, verbs, folds = load_trials(data)
    predictions = _load_predictions(heads, families)
    positive, total = counts[..., 0], counts[..., 1]
    strata = [np.flatnonzero(verbs == verb) for verb in VERB_TYPES]
    observed_shifts = np.diff(positive / total, axis=-1).squeeze(-1)
    human_group_shifts = np.array([observed_shifts[group].mean() for group in strata])
    human_interaction = float(human_group_shifts[1] - human_group_shifts[0])

    condition_losses = positive * (1 - predictions)**2 + (total - positive) * predictions**2
    family_losses = condition_losses.sum(axis=-1)
    family_totals = total.sum(axis=-1)
    brier = family_losses.sum(axis=-1) / family_totals.sum()
    condition_brier = condition_losses.sum(axis=-2) / total.sum(axis=0)

    shifts = predictions[..., 1] - predictions[..., 0]
    shift_errors = shifts - observed_shifts
    group_shifts = np.stack([shifts[..., group].mean(axis=-1) for group in strata], axis=-1)
    interactions = group_shifts[..., 1] - group_shifts[..., 0]
    interaction_errors = interactions - human_interaction
    absolute_interaction_errors = np.abs(interaction_errors)
    brier_samples, interaction_samples = _paired_intervals(
        family_losses[1] - family_losses[0], family_totals, shift_errors[:2], strata,
    )

    arms = {}
    for index, arm in enumerate(ARMS):
        arms[arm] = {
            "brier": _summarize(brier[index]),
            "condition_brier": {
                condition: _summarize(condition_brier[index, :, c])
                for c, condition in enumerate(CONDITIONS)
            },
            "mean_absolute_item_shift_error": _summarize(np.abs(shift_errors[index]).mean(axis=-1)),
            "subject_choice_shift_by_verb_type": {
                verb: _summarize(group_shifts[index, :, g]) for g, verb in enumerate(VERB_TYPES)
            },
            "interaction_ic2_minus_ic1": _summarize(interactions[index]),
            "interaction_error_vs_human": _summarize(interaction_errors[index]),
            "absolute_interaction_error": _summarize(absolute_interaction_errors[index]),
        }

    sources = {name: data / name for name in ("stimuli.jsonl", "trials.jsonl", "splits.json")}
    sources["heads.npz"] = heads
    metrics = {
        "experiment": "Zhan et al. (2020), Experiment 1, eligible continuation choices",
        "conditions": list(CONDITIONS),
        "label_draw_seeds": list(SEEDS),
        "observed_trials": int(total.sum()),
        "stimulus_families": len(families),
        "folds": len(set(folds)),
        "scope": "Held-out stimulus families with overlapping readers. Metrics are means of scores per label draw.",
        "shift_definition": "Within each family, the subject-choice probability under pronoun minus free prompts.",
        "interaction_definition": "The equal-family-weight mean shift for IC-2 minus the mean shift for IC-1.",
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
        "human_by_verb_type": {
            verb: {
                "families": len(group),
                "trials_by_condition": total[group].sum(axis=0).tolist(),
                "subject_choices_by_condition": positive[group].sum(axis=0).tolist(),
                "mean_item_shift": float(human_group_shifts[g]),
            }
            for g, (verb, group) in enumerate(zip(VERB_TYPES, strata, strict=True))
        },
        "human_interaction_ic2_minus_ic1": human_interaction,
        "arms": arms,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "resampling": "Paired stimulus families within verb type. Reader panel, label draws and fitted predictions fixed.",
        },
        "weak_minus_limited": {
            "brier": _contrast(brier[1] - brier[0], brier_samples),
            "absolute_interaction_error": _contrast(
                absolute_interaction_errors[1] - absolute_interaction_errors[0], interaction_samples,
            ),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    mean_probabilities = predictions.mean(axis=1)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, family in enumerate(families):
            row = {
                "family_id": family,
                "verb_type": str(verbs[index]),
                "fold": int(folds[index]),
                "conditions": list(CONDITIONS),
                "subject_choices": positive[index].tolist(),
                "trials": total[index].tolist(),
                "subject_probabilities_mean_over_label_draws": {
                    arm: mean_probabilities[a, index].tolist() for a, arm in enumerate(ARMS)
                },
            }
            handle.write(json.dumps(row, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--heads", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.data, args.heads, args.output)
