"""Cross-fit Frick first-report hazards with an auxiliary four-phase likelihood."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.semantic_tracking import frick_consistency_score as base
from research.semantic_tracking import frick_pair_evaluate as paired_evaluation
from research.semantic_tracking import frick_pair_score as paired
from research.semantic_tracking.rodger_belief_readout import standardization


PROTOCOL = Path(__file__).with_name("frick-mass-calibration-protocol.md")
GRID = (0.0, 0.25, 1.0, 4.0)
INNER_FOLDS = 4
MARGIN_NLL = 0.05
METHODS = ("calibrated", "global", "paired_effect", "original_contextual")
PHASES = paired_evaluation.PHASES


class PhaseGroups(NamedTuple):
    index: np.ndarray
    before: np.ndarray
    at: np.ndarray
    after: np.ndarray
    observed: np.ndarray
    trial_counts: np.ndarray


def phase_groups(paths, hits, totals):
    width = max(len(rows) for rows, _ in paths)
    index = np.zeros((len(paths), width), dtype=np.intp)
    before, at, after = (np.zeros_like(index, dtype=bool) for _ in range(3))
    observed = np.empty((len(paths), 4), dtype=np.float64)
    trial_counts = np.empty(len(paths), dtype=np.float64)
    for i, (rows, target) in enumerate(paths):
        if not 0 < target < len(rows) - 1:
            raise ValueError("Each Frick path must be observed before and after its target")
        index[i, :len(rows)] = rows
        before[i, :target] = True
        at[i, target] = True
        after[i, target + 1:len(rows)] = True
        first = hits[rows]
        readers = totals[rows[0]]
        if readers <= 0 or not np.array_equal(totals[rows], readers - np.r_[0, np.cumsum(first[:-1])]):
            raise ValueError("First-report risk counts do not match complete reader paths")
        observed[i] = first[:target].sum(), first[target], first[target + 1:].sum(), readers - first.sum()
        trial_counts[i] = readers
    return PhaseGroups(index, before, at, after, observed / trial_counts[:, None], trial_counts)


def phase_loss(logits, groups):
    """Return mean observed phase cross-entropy and its gradient over sentence logits."""
    index, before, at, after, observed, _ = groups
    z = logits[index]
    log_survival = -np.logaddexp(0, z)
    prior = (log_survival * before).sum(axis=1)
    at_survival = (log_survival * at).sum(axis=1)
    later = (log_survival * after).sum(axis=1)
    event_before = -np.expm1(prior)
    event_after = -np.expm1(later)
    log_probabilities = np.column_stack((
        np.log(event_before),
        prior + (-np.logaddexp(0, -z) * at).sum(axis=1),
        prior + at_survival + np.log(event_after),
        prior + at_survival + later,
    ))
    hazards = expit(z)
    q_before, q_at, q_after, q_none = observed.T
    gradient = (
        before * ((q_at + q_after + q_none - q_before * np.exp(prior) / event_before)[:, None] * hazards)
        + at * ((q_after + q_none)[:, None] * hazards - q_at[:, None] * expit(-z))
        + after * ((q_none - q_after * np.exp(later) / event_after)[:, None] * hazards)
    )
    result = np.zeros_like(logits)
    np.add.at(result, index, gradient / len(observed))
    return -float((observed * log_probabilities).sum(axis=1).mean()), result


def objective(parameters, design, hits, misses, phases, coefficient):
    loss, gradient = base.binary_loss(parameters, design, hits, misses)
    if coefficient == 0:
        return loss, gradient
    auxiliary, slope = phase_loss(parameters[0] + design @ parameters[1:], phases)
    gradient[0] += coefficient * slope.sum()
    gradient[1:] += coefficient * (design.T @ slope)
    return loss + coefficient * auxiliary, gradient


def fit(design, hits, totals, phases, coefficient):
    positives, total = int(hits.sum()), int(totals.sum())
    keep = totals > 0
    if not 0 < positives < total:
        raise ValueError("Expected both first-report outcomes in each training fold")
    _, center, scale = standardization(design[keep], totals[keep] / total, ())
    standardized = (design - center) / scale
    initial = np.zeros(1 + design.shape[1])
    initial[0] = np.log(positives) - np.log(total - positives)
    result = minimize(
        objective, initial, jac=True, method="L-BFGS-B",
        args=(standardized, hits / total, (totals - hits) / total, phases, coefficient),
        options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9},
    )
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"First report calibration failed to converge: {result.message}")
    return {"center": center, "scale": scale, "intercept": float(result.x[0]),
            "coefficients": result.x[1:], "iterations": int(result.nit),
            "objective": float(result.fun), "train_at_risk_decisions": total}


def indices_for(keys, slots, size):
    seen = np.zeros(size, dtype=bool)
    for key in keys:
        seen[slots[key]] = True
    return np.flatnonzero(seen)


def local_phases(keys, indices, metadata, slots, hits, totals):
    where = np.full(len(hits), -1, dtype=np.intp)
    where[indices] = np.arange(len(indices))
    paths = [(where[slots[key]], metadata[key]["target_index"] - 1) for key in sorted(keys)]
    if any(np.any(rows < 0) for rows, _ in paths):
        raise ValueError("Every complete observed path must be included in its assigned fold")
    return phase_groups(paths, hits[indices], totals[indices])


def nested_fit(states, lexical, families, metadata, slots, hits, totals, frozen_global):
    size = len(hits)
    predictions = np.full(size, np.nan)
    receipts = []
    all_stories = tuple(sorted(families))

    def keys(stories):
        return [key for story in stories for key in families[story].values()]

    def design(stories, outer):
        # The existing projector takes an exclusion key. Only training families
        # and that single excluded key are passed to it, including in inner fits.
        training = {story: families[story] for story in (*stories, outer)}
        axis = paired.paired_direction(states, metadata, training, excluded=outer)
        return np.vstack([np.column_stack((lexical[key], paired.project_story(states[key], axis))) for key in slots])

    def training_parts(stories):
        keys_in_fold = keys(stories)
        ids = indices_for(keys_in_fold, slots, size)
        return ids, local_phases(keys_in_fold, ids, metadata, slots, hits, totals)

    for outer in all_stories:
        development = [story for story in all_stories if story != outer]
        cv = np.zeros((len(GRID), 2), dtype=np.float64)
        memberships = []
        for fold in range(INNER_FOLDS):
            validation = development[fold::INNER_FOLDS]
            inner_train = [story for story in development if story not in validation]
            x = design(inner_train, outer)
            train_ids, train_groups = training_parts(inner_train)
            val_ids, val_groups = training_parts(validation)
            memberships.append(validation)
            for i, coefficient in enumerate(GRID):
                fitted = fit(x[train_ids], hits[train_ids], totals[train_ids], train_groups, coefficient)
                pred = base.predict_logits(x[val_ids], fitted)
                coarse, _ = phase_loss(pred, val_groups)
                step_nll = np.logaddexp(0, pred) * totals[val_ids] - pred * hits[val_ids]
                active = val_groups.before | val_groups.at | val_groups.after
                full = np.sum(step_nll[val_groups.index] * active, axis=1) / val_groups.trial_counts
                cv[i] += np.array((coarse, full.mean())) * len(validation) / len(development)
        eligible = [i for i, (_, nll) in enumerate(cv) if nll <= cv[0, 1] + MARGIN_NLL]
        best = min(eligible, key=lambda i: (cv[i, 0], GRID[i]))
        x = design(development, outer)
        train_ids, train_groups = training_parts(development)
        outer_ids = indices_for(keys((outer,)), slots, size)
        fitted = fit(x[train_ids], hits[train_ids], totals[train_ids], train_groups, GRID[best])
        predictions[outer_ids] = base.predict_logits(x[outer_ids], fitted)
        if GRID[best] == 0 and not np.allclose(
            predictions[outer_ids], frozen_global[outer_ids], atol=1e-9, rtol=1e-8,
        ):
            raise ValueError("The zero-auxiliary fit differs from the frozen global readout")
        receipts.append({
            "held_out_story": outer,
            "selected_lambda": GRID[best],
            "inner_validation_families": memberships,
            "inner_scores": [
                {"lambda": coefficient, "phase_cross_entropy": float(phase), "exact_first_event_nll": float(nll),
                 "eligible": i in eligible} for i, (coefficient, (phase, nll)) in enumerate(zip(GRID, cv, strict=True))
            ],
            "fit": {key: val.tolist() if isinstance(val, np.ndarray) else val for key, val in fitted.items()},
        })
    if not np.isfinite(predictions).all():
        raise ValueError("Each scored stop needs an outer-fold prediction")
    return predictions, receipts


def sealed_comparators(previous, original, slots, stimuli, provenance):
    old_path = previous / "forecast-seal.json"
    old_seal = base.read_json(old_path)
    prior_path = original / "forecast-seal.json"
    prior_seal = base.read_json(prior_path)
    for source, seal in ((previous, old_seal), (original, prior_seal)):
        if (base.digest(source / "forecasts.jsonl") != seal["file_sha256"]["forecasts.jsonl"]
                or any(seal["provenance"][key] != provenance[key] for key in (
                    "state_extraction_sha256", "source_audit_sha256",
                ))):
            raise ValueError("Frozen comparator forecasts do not match the same source and cached states")
    size = sum(slot.stop - slot.start for slot in slots.values())
    previous_firsts = np.empty((2, size))
    expected = ((key, stop) for key in slots for stop in stimuli[key]["stops"][1:])
    for i, (row, identity) in enumerate(zip(base.read_rows(previous / "forecasts.jsonl"), expected, strict=True)):
        if (row["id"], row["stop"]) != identity:
            raise ValueError("Frozen paired comparator coordinates differ")
        for m, method in enumerate(("global", "paired_effect")):
            item = row["methods"][method]
            previous_firsts[m, i] = item["first_logit"]
            if expit(item["first_logit"]) != item["first_d_hazard_probability"]:
                raise ValueError("Frozen paired comparator logits are inconsistent")
    original_first = base.read_forecasts(original / "forecasts.jsonl", slots, stimuli)[
        base.TASKS.index("first"), base.METHODS.index("contextual")
    ]
    return np.vstack((previous_firsts, original_first)), {
        "paired_previous_seal_sha256": base.digest(old_path), "original_seal_sha256": base.digest(prior_path),
    }


def outcome_tables(logits, slots, families, metadata, trials, readers):
    n_stories = len(families)
    shape = (n_stories, len(base.CONDITIONS), len(readers))
    nll = np.zeros((*shape, len(METHODS)))
    human = np.zeros((*shape, len(PHASES)), dtype=np.int8)
    model = np.empty((*shape[:2], len(METHODS), len(PHASES)))
    reader_id = {reader: i for i, reader in enumerate(readers)}
    story_id = {story: i for i, story in enumerate(sorted(families))}
    variants = {}
    for key, slot in slots.items():
        info = metadata[key]
        s, c = story_id[info["story"]], base.CONDITIONS.index(info["condition"])
        target = info["target_index"] - 1
        full = base.first_distribution(logits[:, slot])
        mass = np.exp(full)
        model[s, c] = np.column_stack((
            mass[:, :target].sum(axis=1), mass[:, target], mass[:, target + 1:-1].sum(axis=1), mass[:, -1],
        ))
        variants[key] = s, c, target, full
    for key, reader, _, first in trials:
        s, c, target, full = variants[key]
        r = reader_id[reader]
        phase = 3 if first is None else int(first >= target) + int(first > target)
        human[s, c, r, phase] = 1
        nll[s, c, r] = -full[:, -1 if first is None else first]
    return human, model, nll


def evaluate(logits, slots, families, metadata, trials, readers):
    human, model, nll = outcome_tables(logits, slots, families, metadata, trials, readers)
    present = human.sum(axis=-1)
    n_stories, n_conditions, n_readers = present.shape
    flat_nll = nll.transpose(0, 1, 3, 2).reshape(-1, n_readers)

    def estimate(reader_weights, story_weights):
        denominator = (reader_weights @ present.reshape(-1, n_readers).T).reshape(
            len(reader_weights), n_stories, n_conditions, 1,
        )
        if np.any(denominator == 0):
            raise ValueError("Bootstrap draw dropped an observed story version")
        sampled_nll = (reader_weights @ flat_nll.T).reshape(
            len(reader_weights), n_stories, n_conditions, len(METHODS),
        ) / denominator
        sampled_human = paired_evaluation.reader_phase_means(reader_weights, present, human)
        weights = story_weights / story_weights.sum(axis=1, keepdims=True)
        return (
            np.einsum("bs,bscm->bm", weights, sampled_nll) / n_conditions,
            np.einsum("bs,bscp->bcp", weights, sampled_human),
            np.einsum("bs,scmp->bcmp", weights, model),
        )

    point = estimate(np.ones((1, n_readers)), np.ones((1, n_stories)))
    rng = np.random.default_rng(base.SEED)
    readers_boot = rng.multinomial(n_readers, np.full(n_readers, 1 / n_readers), size=base.DRAWS)
    stories_boot = rng.multinomial(n_stories, np.full(n_stories, 1 / n_stories), size=base.DRAWS)
    boot_chunks = [estimate(readers_boot[i:i + 200], stories_boot[i:i + 200]) for i in range(0, base.DRAWS, 200)]
    bootstrap = tuple(np.concatenate(items) for items in zip(*boot_chunks, strict=True))

    def stats(select):
        return {"estimate": float(select(*point).item()), "crossed_story_reader_ci95": base.interval(select(*bootstrap))}

    def target_effect(prob):
        return prob[:, 1, 1] - prob[:, 0, 1]

    human_effect = stats(lambda _, h, __: target_effect(h))
    methods = {}
    for m, method in enumerate(METHODS):
        methods[method] = {
            "exact_first_event_nll": stats(lambda d, _, __: d[:, m]),
            "target_effect": stats(lambda _, __, p: target_effect(p[:, :, m])),
            "target_model_minus_human": stats(lambda _, h, p: target_effect(p[:, :, m]) - target_effect(h)),
        }
    phase_summary = {
        condition: {
            "human_counts": dict(zip(PHASES, human[:, c].sum(axis=(0, 1)).tolist(), strict=True)),
            "human_probabilities": {phase: stats(lambda _, h, __: h[:, c, p]) for p, phase in enumerate(PHASES)},
            "methods": {method: {
                "probabilities": {phase: stats(lambda _, __, m: m[:, c, j, p]) for p, phase in enumerate(PHASES)},
                "model_minus_human": {
                    phase: stats(lambda _, h, m: m[:, c, j, p] - h[:, c, p]) for p, phase in enumerate(PHASES)
                },
            } for j, method in enumerate(METHODS)},
        } for c, condition in enumerate(base.CONDITIONS)
    }
    comparisons = {
        method: stats(lambda d, _, __: d[:, 0] - d[:, i]) for i, method in enumerate(METHODS[1:], 1)
    }

    def inside_margin(metric):
        lo, hi = metric["crossed_story_reader_ci95"]
        return -base.MARGIN < lo and hi < base.MARGIN

    calibrated = methods["calibrated"]
    gates = {
        "target_effect_positive": calibrated["target_effect"]["crossed_story_reader_ci95"][0] > 0,
        "target_effect_equivalent": inside_margin(calibrated["target_model_minus_human"]),
        **{f"first_event_nll_noninferior_to_{key}": comparisons[key]["crossed_story_reader_ci95"][1] < MARGIN_NLL
           for key in ("global", "original_contextual")},
        **{f"pretarget_mass_equivalent_{condition}": inside_margin(
            phase_summary[condition]["methods"]["calibrated"]["model_minus_human"]["before_target"],
        ) for condition in base.CONDITIONS},
    }
    return {
        "human_target_effect": human_effect,
        "methods": methods,
        "calibrated_minus_comparator_nll": comparisons,
        "first_d_by_condition": phase_summary,
        "gate_results": gates,
        "pass": all(gates.values()),
        "bootstrap": {"draws": base.DRAWS, "seed": base.SEED, "probability_margin": base.MARGIN,
                      "nll_margin_nats_per_story_trial": MARGIN_NLL},
    }


def run(assay, extraction, packet, previous, original, output):
    if output.exists():
        raise FileExistsError(output)
    protocol_sha = base.digest(PROTOCOL)
    audit, stimuli, metadata, families, texts = base.load_layout(assay)
    states, provenance = base.load_states(assay, extraction, packet, base.PROTOCOL, stimuli, metadata, families)
    normalized = {key: paired.normalize_states(state) for key, state in states.items()}
    lexical = {key: paired.sentence_forms(texts[key]) for key in sorted(metadata)}
    stops = np.r_[0, np.cumsum([len(rows) for rows in lexical.values()])]
    slots = {key: slice(start, stop) for key, start, stop in zip(lexical, stops[:-1], stops[1:], strict=True)}
    comparators, comparator_seals = sealed_comparators(previous, original, slots, stimuli, provenance)
    trials, readers, hits, totals, inventory = base.load_outcomes(assay, audit, metadata, slots)
    predicted, fit_receipt = nested_fit(normalized, lexical, families, metadata, slots, hits[1], totals[1], comparators[0])
    logits = np.vstack((predicted, comparators))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".frick-mass-") as temp:
        stage = Path(temp) / "results"
        stage.mkdir()
        forecasts = stage / "forecasts.jsonl"
        with forecasts.open("x", encoding="utf-8") as stream:
            for key, slot in slots.items():
                for raw, stop in zip(logits[:, slot].T, stimuli[key]["stops"][1:], strict=True):
                    record = {"id": key, "stop": stop, "first_logits": dict(zip(METHODS, raw.tolist(), strict=True))}
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        base.write_json(stage / "fits.json", {"features": paired.FEATURES, "lambda_grid": GRID, "folds": fit_receipt})
        seal = stage / "forecast-seal.json"
        base.write_json(seal, {
            "provenance": provenance,
            "protocol_sha256": protocol_sha,
            "code_sha256": base.digest(Path(__file__)),
            **comparator_seals,
            "file_sha256": {name: base.digest(stage / name) for name in ("fits.json", "forecasts.jsonl")},
        })
        restored = np.array([[row["first_logits"][name] for row in base.read_rows(forecasts)] for name in METHODS])
        if not np.array_equal(logits, restored):
            raise ValueError("Held-out scores changed during forecast serialization")
        report = evaluate(restored, slots, families, metadata, trials, readers)
        base.write_json(stage / "results.json", {
            **report, "inventory": inventory, "retrospective": True, "forecast_seal_sha256": base.digest(seal),
        })
        stage.rename(output)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for option in ("assay", "extraction", "packet", "output"):
        cli.add_argument(f"--{option}", type=Path, required=True)
    cli.add_argument("--previous", type=Path, default=paired.REFERENCE.parent / "frick-pair")
    cli.add_argument("--original", type=Path, default=paired.REFERENCE)
    run(**vars(cli.parse_args()))
