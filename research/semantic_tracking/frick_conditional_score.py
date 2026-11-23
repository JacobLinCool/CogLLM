"""Nested family cross-fitting and retrospective scoring for Frick matched targets."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from typing import NamedTuple

import numpy as np

from research.semantic_tracking import frick_conditional_fit as conditional
from research.semantic_tracking import frick_consistency_score as base
from research.semantic_tracking import frick_mass_calibration as mass
from research.semantic_tracking import frick_pair_evaluate as paired_evaluation
from research.semantic_tracking import frick_pair_score as paired


PROTOCOL = Path(__file__).with_name("frick-conditional-protocol.md")
REFERENCE = paired.REFERENCE.parent / "frick-mass-calibration"
GRID = (0.0, 0.25, 1.0, 4.0)
METHODS = ("conditional", "phase_calibrated", "global", "original_contextual")
PHASES = paired_evaluation.PHASES


class Fold(NamedTuple):
    train: np.ndarray
    validate: np.ndarray
    hits: np.ndarray
    totals: np.ndarray
    val_hits: np.ndarray | None
    val_totals: np.ndarray | None
    training_phases: mass.PhaseGroups
    validation_phases: mass.PhaseGroups | None
    pairs: conditional.TargetPairs
    compared_conditions: np.ndarray | None
    validation_stories: tuple


def validation_metrics(logits, fold):
    """Equal version coarsened and exact scores, plus signed mean probability gaps."""
    groups = fold.validation_phases
    if groups is None:
        raise ValueError("Only the inner validation folds carry held-out counts for tuning")
    index, before, at, after = groups[:4]
    active = before | at | after
    step_nll = np.logaddexp(0, logits) * fold.val_totals - logits * fold.val_hits
    exact = (step_nll[index] * active).sum(axis=1) / groups.trial_counts
    phase_ce, _ = mass.phase_loss(logits, groups)
    prefix_survival = np.exp((-np.logaddexp(0, logits[index]) * before).sum(axis=1))
    target_mass = prefix_survival * np.exp((-np.logaddexp(0, -logits[index]) * at).sum(axis=1))
    errors = np.column_stack((1 - prefix_survival - groups.observed[:, 0], target_mass - groups.observed[:, 1]))
    by_condition = np.stack([errors[fold.compared_conditions == c].mean(axis=0) for c in base.CONDITIONS])
    return np.array([phase_ce, exact.mean(), by_condition[1, 1] - by_condition[0, 1],
                     by_condition[0, 0], by_condition[1, 0]])


def make_fold(outer, train_families, test_families, states, lexical, families, metadata, slots, hits, totals,
              *, inner_validation):
    group = {story: families[story] for story in (*train_families, outer)}
    axis = paired.paired_direction(states, metadata, group, excluded=outer)
    design = np.vstack([np.column_stack((lexical[key], paired.project_story(states[key], axis))) for key in slots])
    train_keys = sorted(key for story in train_families for key in families[story].values())
    test_keys = sorted(key for story in test_families for key in families[story].values())
    train = mass.indices_for(train_keys, slots, len(hits))
    test = mass.indices_for(test_keys, slots, len(hits))
    if len(np.intersect1d(train, test)):
        raise ValueError("Matched material crosses the inner or outer family boundary")
    training_positions = np.full(len(hits), -1, dtype=np.intp)
    training_positions[train] = np.arange(len(train))
    left, right = [], []
    for story in train_families:
        a, b = (families[story][c] for c in base.CONDITIONS)
        ta, tb = (metadata[k]["target_index"] for k in (a, b))
        if ta != tb:
            raise ValueError("Target ordinals must match in each aligned pair")
        left.append(training_positions[slots[a].start + ta - 1])
        right.append(training_positions[slots[b].start + tb - 1])
    pairs = conditional.target_pairs(left, right, hits[train], totals[train])
    test_hits = hits[test] if inner_validation else None
    test_totals = totals[test] if inner_validation else None
    test_phases = mass.local_phases(test_keys, test, metadata, slots, hits, totals) if inner_validation else None
    conditions = np.asarray([metadata[key]["condition"] for key in test_keys]) if inner_validation else None
    return Fold(
        design[train], design[test], hits[train], totals[train], test_hits, test_totals,
        mass.local_phases(train_keys, train, metadata, slots, hits, totals), test_phases, pairs,
        conditions, tuple(test_families),
    ), test


def fit_phase(fold, weight):
    return mass.fit(fold.train, fold.hits, fold.totals, fold.training_phases, weight)


def fit_conditional(fold, lambda_weight, prior, gamma_weight):
    return conditional.fit(
        fold.train, fold.hits, fold.totals, fold.training_phases, lambda_weight, fold.pairs, gamma_weight, prior,
    )


def tune(folds):
    development = sum(len(fold.validation_stories) for fold in folds)
    phase_receipts = []
    phase_metrics = []
    for weight in GRID:
        fits = [fit_phase(fold, weight) for fold in folds]
        metrics = sum(validation_metrics(base.predict_logits(fold.validate, fitted), fold)
                      * len(fold.validation_stories) / development
                      for fold, fitted in zip(folds, fits, strict=True))
        phase_receipts.append(fits)
        phase_metrics.append(metrics)
    global_nll = phase_metrics[0][1]
    eligible_phase = [i for i, scores in enumerate(phase_metrics) if scores[1] <= global_nll + mass.MARGIN_NLL]
    li = min(eligible_phase, key=lambda i: (phase_metrics[i][0], GRID[i]))
    chosen_phase_nll = phase_metrics[li][1]
    gamma_metrics = [phase_metrics[li]]
    for weight in GRID[1:]:
        metrics = sum(
            validation_metrics(base.predict_logits(fold.validate, fit_conditional(fold, GRID[li], prior, weight)), fold)
            * len(fold.validation_stories) / development
            for fold, prior in zip(folds, phase_receipts[li], strict=True)
        )
        gamma_metrics.append(metrics)
    eligible_gamma = [
        i for i, row in enumerate(gamma_metrics) if row[1] <= min(global_nll, chosen_phase_nll) + mass.MARGIN_NLL
        and np.all(np.abs(row[3:]) <= base.MARGIN)
    ]
    gi = min(eligible_gamma, key=lambda i: (abs(gamma_metrics[i][2]), GRID[i])) if eligible_gamma else 0

    def records(weights, measures, eligible, label):
        fields = ("phase_cross_entropy", "first_event_nll", "target_effect_bias", "pretarget_bias_01", "pretarget_bias_02")
        return [{label: weight, "eligible": i in eligible, **dict(zip(fields, scores.tolist(), strict=True))}
                for i, (weight, scores) in enumerate(zip(weights, measures, strict=True))]

    return GRID[li], GRID[gi], {
        "phase_grid": records(GRID, phase_metrics, eligible_phase, "lambda"),
        "conditional_grid": records(GRID, gamma_metrics, eligible_gamma, "gamma"),
        "no_candidate_meets_conditional_selection_constraints": not eligible_gamma,
    }


def crossfit(states, lexical, families, metadata, slots, hits, totals, phase_forecasts):
    data = states, lexical, families, metadata, slots, hits, totals
    outputs = np.full(len(hits), np.nan)
    fits = []
    for outer in sorted(families):
        training = tuple(story for story in sorted(families) if story != outer)
        folds = []
        for index in range(mass.INNER_FOLDS):
            validation = training[index::mass.INNER_FOLDS]
            train_inner = tuple(story for story in training if story not in validation)
            fold, _ = make_fold(outer, train_inner, validation, *data, inner_validation=True)
            folds.append(fold)
        lambda_weight, gamma_weight, receipt = tune(folds)
        fold, outer_indices = make_fold(outer, training, (outer,), *data, inner_validation=False)
        original = fit_phase(fold, lambda_weight)
        phase_result = base.predict_logits(fold.validate, original)
        if not np.allclose(phase_result, phase_forecasts[outer_indices], rtol=1e-8, atol=1e-9):
            raise ValueError("Independent nested phase-only reconstruction differs from the sealed comparator")
        fitted = fit_conditional(fold, lambda_weight, original, gamma_weight) if gamma_weight else original
        outputs[outer_indices] = base.predict_logits(fold.validate, fitted)
        fits.append({
            "held_out_story": outer, "selected_lambda": lambda_weight, "selected_gamma": gamma_weight,
            "original_phase_max_delta": float(np.max(np.abs(phase_result - phase_forecasts[outer_indices]))),
            "matched_training_pairs": len(fold.pairs.left),
            "inner_validation_stories": [part.validation_stories for part in folds],
            "fitted": {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in fitted.items()},
            **receipt,
        })
    if not np.isfinite(outputs).all():
        raise ValueError("Every sentence needs one held-out conditional reading forecast")
    return outputs, fits


def frozen_comparators(reference, slots, stimuli, provenance):
    seal_path = reference / "forecast-seal.json"
    seal = base.read_json(seal_path)
    score = base.read_json(reference / "results.json")
    path = reference / "forecasts.jsonl"
    if (base.digest(path) != seal["file_sha256"]["forecasts.jsonl"]
            or base.digest(seal_path) != score["forecast_seal_sha256"]
            or any(seal["provenance"][key] != provenance[key] for key in ("source_audit_sha256", "state_extraction_sha256"))):
        raise ValueError("The sealed phase forecasts do not come from the same source and extracted states")
    expect = ((key, stop) for key in slots for stop in stimuli[key]["stops"][1:])
    saved = []
    for row, (key, stop) in zip(base.read_rows(path), expect, strict=True):
        if row["id"] != key or row["stop"] != stop:
            raise ValueError("Expected frozen phase, global, and contextual logits at exact aligned sentence ends")
        saved.append([row["first_logits"][method] for method in ("calibrated", "global", "original_contextual")])
    values = np.asarray(saved, dtype=np.float64).T
    if not np.isfinite(values).all():
        raise ValueError("Comparator logits are invalid")
    return values, base.digest(seal_path)


def evaluate(logits, slots, families, metadata, trials, readers):
    human, model, nll = mass.outcome_tables(logits, slots, families, metadata, trials, readers)
    present = human.sum(axis=-1)
    n_stories, n_conditions, n_readers = present.shape
    flattened = nll.transpose(0, 1, 3, 2).reshape(-1, n_readers)

    def estimate(reader_weights, story_weights):
        denominator = (reader_weights @ present.reshape(-1, n_readers).T).reshape(
            len(reader_weights), n_stories, n_conditions, 1,
        )
        if np.any(denominator <= 0):
            raise ValueError("A bootstrap sample dropped readers in a story condition")
        sampled = (reader_weights @ flattened.T).reshape(
            len(reader_weights), n_stories, n_conditions, len(METHODS),
        ) / denominator
        human_rates = paired_evaluation.reader_phase_means(reader_weights, present, human)
        weights = story_weights / story_weights.sum(axis=1, keepdims=True)
        return (np.einsum("bs,bscm->bm", weights, sampled) / n_conditions,
                np.einsum("bs,bscp->bcp", weights, human_rates),
                np.einsum("bs,scmp->bcmp", weights, model))

    point = estimate(np.ones((1, n_readers)), np.ones((1, n_stories)))
    rng = np.random.default_rng(base.SEED)
    reader_draws = rng.multinomial(n_readers, np.full(n_readers, 1 / n_readers), size=base.DRAWS)
    story_draws = rng.multinomial(n_stories, np.full(n_stories, 1 / n_stories), size=base.DRAWS)
    chunks = [estimate(reader_draws[i:i + 200], story_draws[i:i + 200]) for i in range(0, base.DRAWS, 200)]
    draws = tuple(np.concatenate(values) for values in zip(*chunks, strict=True))

    def stat(select):
        return {"estimate": float(select(*point).item()), "crossed_story_reader_ci95": base.interval(select(*draws))}

    def effect(values):
        return values[:, 1, 1] - values[:, 0, 1]

    methods = {}
    for m, name in enumerate(METHODS):
        methods[name] = {
            "exact_first_event_nll": stat(lambda loss, _, __: loss[:, m]),
            "target_effect": stat(lambda _, __, p: effect(p[:, :, m])),
            "target_model_minus_human": stat(lambda _, h, p: effect(p[:, :, m]) - effect(h)),
            "first_d_by_condition": {
                condition: {
                    "predicted": {label: stat(lambda _, __, p: p[:, c, m, index])
                                  for index, label in enumerate(PHASES)},
                    "prediction_minus_human": {label: stat(lambda _, h, p: p[:, c, m, index] - h[:, c, index])
                                               for index, label in enumerate(PHASES)},
                } for c, condition in enumerate(base.CONDITIONS)
            },
        }
    compared = {
        name: {"conditional_minus_comparator_nll": stat(lambda nll, _, __: nll[:, 0] - nll[:, m]),
               "conditional_minus_comparator_target_effect": stat(
                   lambda _, __, p: effect(p[:, :, 0]) - effect(p[:, :, m]),
               ),
               "conditional_target_absolute_error_reduction": stat(
                   lambda _, h, p: np.abs(effect(p[:, :, m]) - effect(h)) - np.abs(effect(p[:, :, 0]) - effect(h)),
               )} for m, name in enumerate(METHODS[1:], 1)
    }

    def equivalent(estimate):
        lo, hi = estimate["crossed_story_reader_ci95"]
        return -base.MARGIN < lo and hi < base.MARGIN

    current = methods["conditional"]
    gates = {
        "target_effect_positive": current["target_effect"]["crossed_story_reader_ci95"][0] > 0,
        "target_effect_equivalent": equivalent(current["target_model_minus_human"]),
        **{f"nll_noninferior_to_{name}": compared[name]["conditional_minus_comparator_nll"][
            "crossed_story_reader_ci95"
        ][1] < mass.MARGIN_NLL for name in ("global", "original_contextual")},
        **{f"pretarget_bias_equivalent_{c}": equivalent(
            current["first_d_by_condition"][c]["prediction_minus_human"]["before_target"],
        ) for c in base.CONDITIONS},
    }
    return {
        "human_target_effect": stat(lambda _, h, __: effect(h)),
        "human_first_d_by_condition": {
            condition: {label: stat(lambda _, h, __: h[:, c, i]) for i, label in enumerate(PHASES)}
            for c, condition in enumerate(base.CONDITIONS)
        },
        "methods": methods, "comparisons": compared, "gate_results": gates, "pass": all(gates.values()),
        "bootstrap": {"draws": base.DRAWS, "seed": base.SEED, "probability_margin": base.MARGIN,
                      "nll_margin_nats_per_story_trial": mass.MARGIN_NLL},
    }


def run(assay, extraction, packet, reference, output):
    if output.exists():
        raise FileExistsError(output)
    protocol_sha = base.digest(PROTOCOL)
    audit, stimuli, metadata, families, texts = base.load_layout(assay)
    states, provenance = base.load_states(assay, extraction, packet, base.PROTOCOL, stimuli, metadata, families)
    family_order = sorted(metadata)
    forms = {key: paired.sentence_forms(texts[key]) for key in family_order}
    offsets = np.r_[0, np.cumsum([len(forms[key]) for key in family_order])]
    slots = {key: slice(lo, hi) for key, lo, hi in zip(family_order, offsets[:-1], offsets[1:], strict=True)}
    for story, pair in families.items():
        a, b = (pair[key] for key in base.CONDITIONS)
        if metadata[a]["target_index"] != metadata[b]["target_index"] or (
            texts[a][metadata[a]["target_index"]] != texts[b][metadata[b]["target_index"]]
        ):
            raise ValueError(f"Expected one aligned literal target for story {story}")
    comparators, prior_seal_hash = frozen_comparators(reference, slots, stimuli, provenance)
    trials, readers, hits, totals, inventory = base.load_outcomes(assay, audit, metadata, slots)
    states = {key: paired.normalize_states(states[key]) for key in family_order}
    fitted, families_fit = crossfit(states, forms, families, metadata, slots, hits[1], totals[1], comparators[0])
    logits = np.vstack((fitted, comparators))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".frick-conditional-") as tmp:
        stage = Path(tmp) / "conditional"
        stage.mkdir()
        forecasts = stage / "forecasts.jsonl"
        with forecasts.open("x", encoding="utf-8") as stream:
            for key, slot in slots.items():
                for stop, row in zip(stimuli[key]["stops"][1:], logits[:, slot].T, strict=True):
                    value = {"id": key, "stop": stop, "first_logits": dict(zip(METHODS, row.tolist(), strict=True))}
                    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        base.write_json(stage / "fits.json", {"features": paired.FEATURES, "grid": GRID, "folds": families_fit})
        seal = stage / "forecast-seal.json"
        base.write_json(seal, {
            "provenance": provenance, "protocol_sha256": protocol_sha,
            "reference_forecast_seal_sha256": prior_seal_hash,
            "scorer_sha256": base.digest(Path(__file__)), "fitter_sha256": base.digest(Path(conditional.__file__)),
            "file_sha256": {name: base.digest(stage / name) for name in ("forecasts.jsonl", "fits.json")},
        })
        restored = np.asarray(
            [[row["first_logits"][m] for m in METHODS] for row in base.read_rows(forecasts)], dtype=np.float64,
        ).T
        if not np.array_equal(restored, logits):
            raise ValueError("Sealed new forecasts must exactly reproduce the outer-fold predictions")
        scores = evaluate(restored, slots, families, metadata, trials, readers)
        base.write_json(stage / "results.json", {
            **scores, "inventory": inventory, "retrospective": True, "forecast_seal_sha256": base.digest(seal),
        })
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("assay", "extraction", "packet", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    run(**vars(parser.parse_args()))
