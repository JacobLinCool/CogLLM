"""Compare the two Stage 4 readouts and score sealed external predictions."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from ..stage3.zhan_landmark_probe import CONDITIONS, VERB_TYPES, load_trials
from .zhan_weak import ARMS, SEEDS, _contrast_coefficients
from .zhan_weak_score import _contrast, _load_predictions, _paired_intervals, _summarize


READOUTS = ("shared", "state_update")
TOKENIZATION = ("boundary", "native")
SPELLINGS = ("ta-m", "ta-f")
EXTERNAL_CSV_SHA256 = "d308975fcecdc4f8ae3b39ef9cc4febb9a3424d0031ca0d8cfe2987ee12f1557"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_source_hashes():
    current = Path(__file__)
    stage3 = current.parent.parent / "stage3"
    return {
        path.name: digest(path)
        for path in (
            current, current.with_name("zhan_weak.py"), current.with_name("zhan_weak_score.py"),
            stage3 / "zhan_landmark_probe.py", stage3 / "issue_probe.py", stage3 / "probe.py",
        )
    }


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def score(predictions, positive, total, verbs):
    strata = [np.flatnonzero(verbs == verb) for verb in VERB_TYPES]
    if any(not len(group) for group in strata) or (total <= 0).any():
        raise ValueError("Every item needs both conditions and both verb strata must be represented")
    rates = positive / total
    human_shifts = rates[:, 1] - rates[:, 0]
    human_interaction = human_shifts[strata[1]].mean() - human_shifts[strata[0]].mean()
    loss = positive * (1 - predictions)**2 + (total - positive) * predictions**2
    brier = loss.sum(axis=(-2, -1)) / total.sum()
    condition_brier = loss.sum(axis=-2) / total.sum(axis=0)
    shift = predictions[..., 1] - predictions[..., 0]
    shift_errors = shift - human_shifts
    group_shifts = np.stack([shift[..., group].mean(axis=-1) for group in strata], axis=-1)
    interaction = group_shifts[..., 1] - group_shifts[..., 0]
    absolute_interaction_errors = np.abs(interaction - human_interaction)
    pre_cue_gap = predictions[..., strata[0], 0].mean(axis=-1) - predictions[..., strata[1], 0].mean(axis=-1)

    methods = {}
    for readout_id, readout in enumerate(READOUTS):
        methods[readout] = {}
        for arm_id, arm in enumerate(ARMS):
            key = (readout_id, arm_id)
            methods[readout][arm] = {
                "brier": _summarize(brier[key]),
                "condition_brier": {
                    condition: _summarize(condition_brier[key][..., index])
                    for index, condition in enumerate(CONDITIONS)
                },
                "pre_cue_gap_ic1_minus_ic2": _summarize(pre_cue_gap[key]),
                "subject_choice_shift_by_verb_type": {
                    verb: _summarize(group_shifts[key][..., index]) for index, verb in enumerate(VERB_TYPES)
                },
                "interaction_ic2_minus_ic1": _summarize(interaction[key]),
                "absolute_interaction_error": _summarize(absolute_interaction_errors[key]),
                "mean_absolute_item_shift_error": _summarize(np.abs(shift_errors[key]).mean(axis=-1)),
            }

    difference = {}
    for arm_id, arm in enumerate(ARMS):
        family_loss_delta = (loss[1, arm_id] - loss[0, arm_id]).sum(axis=-1)
        samples_brier, samples_interaction = _paired_intervals(
            family_loss_delta, total.sum(axis=1), shift_errors[:, arm_id], strata,
        )
        difference[arm] = {
            "brier": _contrast(brier[1, arm_id] - brier[0, arm_id], samples_brier),
            "absolute_interaction_error": _contrast(
                absolute_interaction_errors[1, arm_id] - absolute_interaction_errors[0, arm_id],
                samples_interaction,
            ),
        }
    return {
        "observed_trials": int(total.sum()),
        "human_trials_by_condition": total.sum(axis=0).tolist(),
        "human_subject_choices_by_condition": positive.sum(axis=0).tolist(),
        "families_by_verb_type": dict(zip(VERB_TYPES, [len(group) for group in strata], strict=True)),
        "human_interaction_ic2_minus_ic1": float(human_interaction),
        "human_pre_cue_gap_ic1_minus_ic2": float(rates[strata[0], 0].mean() - rates[strata[1], 0].mean()),
        "human_subject_choice_shift_by_verb_type": {
            verb: float(human_shifts[group].mean()) for verb, group in zip(VERB_TYPES, strata, strict=True)
        },
        "readouts": methods,
        "state_update_minus_shared": difference,
    }


def score_zhan(data, shared, updated, output):
    families, _, counts, verbs, _ = load_trials(data)
    predictions = np.stack([_load_predictions(path, families) for path in (shared, updated)])
    save_json(output, {
        "scope": "Zhan Experiment 1, five fixed held-out material folds with overlapping readers.",
        "input_sha256": {key: digest(path) for key, path in (("shared", shared), ("state_update", updated))},
        **score(predictions, counts[..., 0], counts[..., 1], verbs),
    })


def _probabilities(hidden, weights):
    design = np.concatenate((hidden, np.ones((*hidden.shape[:-1], 1))), axis=-1)
    logits = np.einsum("adk,fck->adfc", weights, design)
    return np.exp(-np.logaddexp(0, -logits))


def _normalized_states(values, name):
    states = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(states, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError(f"Invalid state norms in {name}")
    return states / norm


def seal_external(features, heads, output):
    with np.load(heads, allow_pickle=False) as models:
        baseline, updated = models["baseline_parameters"], models["update_parameters"]
        annot = (models["arms"].tolist(), models["seeds"].tolist())
    if (
        baseline.shape != (len(ARMS), len(SEEDS), 2049)
        or updated.shape != (len(ARMS), len(SEEDS), 4097)
        or annot != (list(ARMS), list(SEEDS))
        or not (np.isfinite(baseline).all() and np.isfinite(updated).all())
    ):
        raise ValueError("Expected the two fixed all-Zhan model sets")

    families = [f"I{index:02d}" for index in range(1, 25)]
    ids = [f"{family}-{prompt}" for family in families for prompt in ("free", *SPELLINGS)]
    probabilities = []
    with np.load(features, allow_pickle=False) as source:
        cached_ids = source["ids"].tolist()
        if len(cached_ids) != len(ids) or set(cached_ids) != set(ids):
            raise ValueError("Expected 24 external items with a free and two overt pronoun texts each")
        take = [cached_ids.index(key) for key in ids]
        prefix = _normalized_states(
            source["controlled_prefix_hidden"][take].reshape(len(families), 3, 2048),
            "controlled_prefix_hidden",
        )
        for key in ("hidden", "native_hidden"):
            states = _normalized_states(source[key][take].reshape(len(families), 3, 2048), key)
            spellings = []
            for column in (1, 2):
                visible = states[:, [0, column]]
                if key == "hidden":
                    visible[:, 0] = prefix[:, column]
                delta = visible - visible[:, :1]
                spellings.append(np.stack([
                    _probabilities(visible, baseline),
                    _probabilities(np.concatenate((visible, delta), axis=-1), updated),
                ]))
            probabilities.append(np.stack(spellings))

    output.mkdir(parents=True, exist_ok=True)
    sealed = output / "external-predictions.npz"
    np.savez_compressed(sealed, families=np.array(families), predictions=np.stack(probabilities))
    save_json(output / "prediction-seal.json", {
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_outcomes_used_to_compute_predictions": False,
        "input_sha256": {key: digest(path) for key, path in (("features", features), ("all_zhan_heads", heads))},
        "predictions_sha256": digest(sealed),
        "score_source_sha256": score_source_hashes(),
        "expected_response_source_sha256": EXTERNAL_CSV_SHA256,
        "tokenization": TOKENIZATION,
        "spellings": SPELLINGS,
        "readouts": READOUTS,
        "arms": ARMS,
        "seeds": SEEDS,
    })


def external_trials(path, families):
    positions = {key: index for index, key in enumerate(families)}
    positive = np.zeros((len(families), len(CONDITIONS)), dtype=np.int64)
    total = np.zeros_like(positive)
    verbs = {}
    readers = set()
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Verb_Type"] != "IC":
                continue
            family = row["ID"]
            key = (row["Subject"], family)
            if key in seen or family not in positions or row["Referent"] not in ("N1", "N2"):
                raise ValueError(f"Unexpected included Lam & Hwang trial for {family}")
            seen.add(key)
            readers.add(row["Subject"])
            verb = VERB_TYPES[("N1", "N2").index(row["Bias"])]
            if verbs.setdefault(family, verb) != verb:
                raise ValueError(f"Inconsistent verb condition for {family}")
            index = (positions[family], CONDITIONS.index(row["Prompt_Type"]))
            total[index] += 1
            positive[index] += row["Referent"] == "N1"
    if set(verbs) != set(families):
        raise ValueError("Expected labels for every eligible IC item")
    return positive, total, np.array([verbs[key] for key in families]), len(readers)


def spelling_bounds(predictions, positive, total, verbs):
    # First average over label draws, so each material uses the same spelling
    # for all fitted readouts within an arm.
    per_item_brier = (
        positive * (1 - predictions)**2 + (total - positive) * predictions**2
    ).mean(axis=-3).sum(axis=-1) / total.sum()
    contrast = _contrast_coefficients(verbs).reshape(-1, 2)
    per_item_effect = (predictions.mean(axis=-3) * contrast).sum(axis=-1)

    def bounds(by_spelling):
        return np.stack((by_spelling.min(axis=0).sum(axis=-1), by_spelling.max(axis=0).sum(axis=-1)), axis=-1)

    return {
        "axis_order": [list(READOUTS), list(ARMS), ["min", "max"]],
        "brier": bounds(per_item_brier).tolist(),
        "interaction_ic2_minus_ic1": bounds(per_item_effect).tolist(),
        "brier_state_update_minus_shared_axis_order": [list(ARMS), ["min", "max"]],
        "brier_state_update_minus_shared": bounds(per_item_brier[:, 1] - per_item_brier[:, 0]).tolist(),
    }


def score_external(sealed, responses, output):
    seal = json.loads((sealed.parent / "prediction-seal.json").read_text(encoding="utf-8"))
    if (
        seal["external_outcomes_used_to_compute_predictions"] or seal["predictions_sha256"] != digest(sealed)
        or seal["score_source_sha256"] != score_source_hashes()
        or seal["expected_response_source_sha256"] != digest(responses)
    ):
        raise ValueError("Frozen prediction, scoring code, or response source does not match its seal")
    with np.load(sealed, allow_pickle=False) as data:
        families = data["families"].tolist()
        predictions = data["predictions"]
    shape = (len(TOKENIZATION), len(SPELLINGS), len(READOUTS), len(ARMS), len(SEEDS), 24, len(CONDITIONS))
    if predictions.shape != shape or not np.all((0 <= predictions) & (predictions <= 1)):
        raise ValueError("Invalid sealed probability inventory")
    positive, total, verbs, readers = external_trials(responses, families)
    save_json(output, {
        "scope": "Lam & Hwang Experiment 1, independent IC items. Model scoring by both fixed overt-spelling scenarios.",
        "participant_count": readers,
        "prediction_seal_sha256": digest(sealed.parent / "prediction-seal.json"),
        "prediction_npz_sha256": digest(sealed),
        "responses_sha256": EXTERNAL_CSV_SHA256,
        "cases": {
            tokenization: {
                "by_spelling": {
                    spelling: score(values, positive, total, verbs)
                    for spelling, values in zip(SPELLINGS, cases, strict=True)
                },
                "any_itemwise_spelling_assignment": spelling_bounds(cases, positive, total, verbs),
            }
            for tokenization, cases in zip(TOKENIZATION, predictions, strict=True)
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function, arguments in (
        ("score-zhan", score_zhan, ("data", "shared", "updated", "output")),
        ("seal-external", seal_external, ("features", "heads", "output")),
        ("score-external", score_external, ("sealed", "responses", "output")),
    ):
        command = commands.add_parser(name)
        for argument in arguments:
            command.add_argument(f"--{argument}", type=Path, required=True)
        command.set_defaults(function=function, arguments=arguments)
    args = parser.parse_args()
    args.function(*(getattr(args, name) for name in args.arguments))


if __name__ == "__main__":
    main()
