"""Compare Frick's seven-feature conditional readout with causal context features."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from research.semantic_tracking import frick_conditional_score as prior
from research.semantic_tracking import frick_consistency_score as base
from research.semantic_tracking import frick_context_features as context
from research.semantic_tracking import frick_mass_calibration as mass
from research.semantic_tracking import frick_pair_score as paired


PROTOCOL = Path(__file__).with_name("frick-context-protocol.md")
REFERENCE = paired.REFERENCE.parent / "frick-conditional"
METHODS = ("context_features", "seven_feature_conditional", "global", "original_contextual")
FEATURES = (*base.SCALAR_FEATURES, *context.FEATURES, *paired.FEATURES[len(base.SCALAR_FEATURES):])


def reference_logits(reference, slots, stimuli, provenance, comparators):
    seal_path = reference / "forecast-seal.json"
    forecast_path = reference / "forecasts.jsonl"
    seal = base.read_json(seal_path)
    score = base.read_json(reference / "results.json")
    if (score["forecast_seal_sha256"] != base.digest(seal_path)
            or seal["file_sha256"]["forecasts.jsonl"] != base.digest(forecast_path)
            or any(seal["provenance"][name] != provenance[name] for name in (
                "state_extraction_sha256", "source_audit_sha256",
            ))):
        raise ValueError("The seven-feature readout does not match the sealed source representations")
    expected = ((key, stop) for key in slots for stop in stimuli[key]["stops"][1:])
    rows = []
    for item, identity in zip(base.read_rows(forecast_path), expected, strict=True):
        if (item["id"], item["stop"]) != identity:
            raise ValueError("Expected each frozen forecast at the same causal stop")
        rows.append([item["first_logits"][method] for method in prior.METHODS])
    exact = np.asarray(rows, dtype=np.float64).T
    if not np.isfinite(exact).all() or not np.array_equal(exact[1:], comparators):
        raise ValueError("Prior phase, global and prefix logits differ in the sealed seven-feature forecasts")
    return exact[0], base.digest(seal_path)


def fit_context(states, lexical, families, metadata, slots, hits, totals):
    """Call the exact existing fold, tuning, phase and conditional fit functions."""
    args = states, lexical, families, metadata, slots, hits, totals
    output = np.full(len(hits), np.nan, dtype=np.float64)
    fitted_families = []
    no_target_max_delta = 0.0
    for outer in sorted(families):
        training = tuple(key for key in sorted(families) if key != outer)
        inner = []
        for i in range(mass.INNER_FOLDS):
            validation = training[i::mass.INNER_FOLDS]
            train_inner = tuple(key for key in training if key not in validation)
            fold, _ = prior.make_fold(outer, train_inner, validation, *args, inner_validation=True)
            inner.append(fold)
        phase_weight, conditional_weight, receipt = prior.tune(inner)
        final, indices = prior.make_fold(outer, training, (outer,), *args, inner_validation=False)
        phase_only = prior.fit_phase(final, phase_weight)
        fitted = (
            prior.fit_conditional(final, phase_weight, phase_only, conditional_weight)
            if conditional_weight else phase_only
        )
        prediction = base.predict_logits(final.validate, fitted)
        hidden_metadata = dict(metadata)
        hidden_hits, hidden_totals = hits.copy(), totals.copy()
        for held_out in families[outer].values():
            hidden_metadata[held_out] = {}
            hidden_hits[slots[held_out]] = -1
            hidden_totals[slots[held_out]] = -1
        no_target, alternate_indices = prior.make_fold(
            outer, training, (outer,), states, lexical, families, hidden_metadata, slots, hidden_hits, hidden_totals,
            inner_validation=False,
        )
        if not np.array_equal(indices, alternate_indices):
            raise ValueError("Changing inaccessible held-out metadata changed forecast rows")
        delta = float(np.max(np.abs(prediction - base.predict_logits(no_target.validate, fitted))))
        if delta > 1e-13:
            raise ValueError("Held-out context forecasts depend on unavailable targets or labels")
        no_target_max_delta = max(no_target_max_delta, delta)
        output[indices] = prediction
        fitted_families.append({
            "held_out_story": outer,
            "selected_lambda": phase_weight,
            "selected_gamma": conditional_weight,
            "inner_validation_stories": [member.validation_stories for member in inner],
            "matched_training_pairs": len(final.pairs.left),
            "fitted": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in fitted.items()},
            **receipt,
        })
    if not np.isfinite(output).all():
        raise ValueError("Every hold-out sentence must have one finite context-feature prediction")
    return output, fitted_families, no_target_max_delta


def evaluate(logits, slots, families, metadata, trials, readers):
    # The earlier scorer has four fixed slots. Only their descriptive names are
    # translated here; the desired paired scores and gates are identical.
    result = prior.evaluate(logits, slots, families, metadata, trials, readers)
    result["methods"] = {new: result["methods"][old] for new, old in zip(METHODS, prior.METHODS, strict=True)}
    result["comparisons"] = {
        new: {key.replace("conditional", "context_features", 1): value
              for key, value in result["comparisons"][old].items()}
        for new, old in zip(METHODS[1:], prior.METHODS[1:], strict=True)
    }
    direct = result["comparisons"]["seven_feature_conditional"]
    gates = {
        "target_absolute_error_reduced_vs_seven": (
            direct["context_features_target_absolute_error_reduction"]["crossed_story_reader_ci95"][0] > 0
        ),
        "nll_noninferior_to_seven": (
            direct["context_features_minus_comparator_nll"]["crossed_story_reader_ci95"][1] < mass.MARGIN_NLL
        ),
    }
    result["ablation_gate_results"] = gates
    result["ablation_pass"] = all(gates.values())
    return result


def run(assay, extraction, packet, phase_reference, reference, output):
    if output.exists():
        raise FileExistsError(output)
    protocol_sha = base.digest(PROTOCOL)
    audit, stimuli, metadata, families, texts = base.load_layout(assay)
    states, provenance = base.load_states(assay, extraction, packet, base.PROTOCOL, stimuli, metadata, families)
    keys = sorted(metadata)
    states = {key: paired.normalize_states(states[key]) for key in keys}
    legacy = {key: paired.sentence_forms(texts[key]) for key in keys}
    endpoints = np.r_[0, np.cumsum([len(rows) for rows in legacy.values()])]
    slots = {key: slice(start, stop) for key, start, stop in zip(keys, endpoints[:-1], endpoints[1:], strict=True)}
    richer = {}
    prefix_count, max_prefix_error = 0, 0.0
    for key in keys:
        extra = context.contextual_features(texts[key], states[key])
        checked, worst = context.verify_prefix_features(texts[key], states[key], extra)
        prefix_count += checked
        max_prefix_error = max(max_prefix_error, worst)
        richer[key] = np.column_stack((legacy[key], extra))
    comparators, phase_hash = prior.frozen_comparators(phase_reference, slots, stimuli, provenance)
    original, conditional_hash = reference_logits(reference, slots, stimuli, provenance, comparators)
    trials, readers, hits, totals, inventory = base.load_outcomes(assay, audit, metadata, slots)
    legacy_replayed, _ = prior.crossfit(states, legacy, families, metadata, slots, hits[1], totals[1], comparators[0])
    if not np.array_equal(legacy_replayed, original):
        raise ValueError("The original seven-feature forecasts failed exact replay under the common pipeline")
    candidate, fits, unavailable_max = fit_context(states, richer, families, metadata, slots, hits[1], totals[1])
    logits = np.vstack((candidate, original, comparators[1:]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".frick-context-") as tmp:
        stage = Path(tmp) / "evaluation"
        stage.mkdir()
        forecasts = stage / "forecasts.jsonl"
        with forecasts.open("x", encoding="utf-8") as stream:
            for key, slot in slots.items():
                for stop, row in zip(stimuli[key]["stops"][1:], logits[:, slot].T, strict=True):
                    record = {"id": key, "stop": stop, "first_logits": dict(zip(METHODS, row.tolist(), strict=True))}
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        base.write_json(stage / "fits.json", {"features": FEATURES, "grid": prior.GRID, "folds": fits})
        base.write_json(stage / "verification.json", {
            "legacy_forecasts_replayed_exactly": True,
            "prefix_forecasts_tested_without_suffix": prefix_count,
            "maximum_causal_feature_change_after_suffix_removal": max_prefix_error,
            "maximum_logit_change_after_removing_held_out_targets_and_counts": unavailable_max,
            "legacy_feature_count": len(paired.FEATURES),
            "contextual_feature_count": len(context.FEATURES),
        })
        names = ("fits.json", "forecasts.jsonl", "verification.json")
        seal = stage / "forecast-seal.json"
        base.write_json(seal, {
            "provenance": provenance,
            "protocol_sha256": protocol_sha,
            "phase_reference_forecast_seal_sha256": phase_hash,
            "conditional_reference_forecast_seal_sha256": conditional_hash,
            "scorer_sha256": base.digest(Path(__file__)),
            "features_sha256": base.digest(Path(context.__file__)),
            "file_sha256": {name: base.digest(stage / name) for name in names},
        })
        restored = np.asarray([[row["first_logits"][method] for method in METHODS]
                               for row in base.read_rows(forecasts)], dtype=np.float64).T
        if not np.array_equal(logits, restored):
            raise ValueError("The sealed forecasts differ from the represented held-out predictions")
        scores = evaluate(restored, slots, families, metadata, trials, readers)
        base.write_json(stage / "results.json", {
            **scores, "inventory": inventory, "retrospective": True, "forecast_seal_sha256": base.digest(seal),
        })
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("assay", "extraction", "packet", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--phase-reference", type=Path, default=prior.REFERENCE)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    run(**vars(parser.parse_args()))
