"""Evaluate sealed crossed-fold StudentAnnotations forecasts locally."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np


DESIGN_PROTOCOL = Path(__file__).with_name("student-annotations-background.md")
FIT_CODE = Path(__file__).with_name("student_annotations_fit.py")
PREPARATION_SHA256 = "e9245766647a5ae24fb5347e39bea4d167da534380e44f7419d4464d7639fe97"
FIT_PROTOCOL_SHA256 = "e203668349fbc1f7b216ed0fafea9d195a3d6e5922a27226cd3e2d0569cd9e8f"
METHODS = ("length_group", "qwen_additive", "qwen_interaction", "blank_background")
GRADES = (1, 2, 3, 4, 5)
PROFILE_CODES = {"german": 0, "german_and_another_language": 1, "another_language": 2}
METRICS = ("nll_nats", "ranked_probability_score")
COMPARISONS = (("qwen_interaction", "qwen_additive"), ("qwen_additive", "length_group"),
               ("qwen_additive", "blank_background"))
FOLDS, TRIALS, READERS, SENTENCES, SENTENCE_GROUPS, PAIRED_SENTENCES = 4, 3160, 192, 1143, 1134, 179
DRAWS, SEED, BATCH, SENTENCE_BATCH = 10_000, 43, 64, 12
EQUIVALENCE = 0.5
TIE_ATOL = 1e-12
LOCAL_FILES = {
    "trial_design": "trial-design.jsonl", "readers": "readers.jsonl", "sentences": "sentences.jsonl",
    "outcomes": "outcomes.jsonl",
}
FEATURE_FILES = {"extraction": "extraction.json", "features": "features.npz", "causality": "causality.json"}


@dataclass(frozen=True)
class TrialDesign:
    trials: tuple[str, ...]
    literal_sentences: np.ndarray
    normalized: np.ndarray
    readers: np.ndarray
    language: np.ndarray
    fold_grid: np.ndarray
    fit_profile_counts: dict
    fit_stimulus_counts: dict


@dataclass(frozen=True)
class EffectDesign:
    cases: np.ndarray
    normalized: np.ndarray
    trial_index: np.ndarray
    readers: np.ndarray
    cells: np.ndarray


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path, key, fields):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            if row.keys() != fields or row[key] in result:
                raise ValueError(f"Invalid schema or duplicate key in {path.name}")
            result[row[key]] = row
    return result


def read_design(local, receipt):
    readers = records(local / LOCAL_FILES["readers"], "reader", {"reader", "fold", "language"})
    sentences = records(local / LOCAL_FILES["sentences"], "id", {"id", "fold", "group"})
    trials = records(local / LOCAL_FILES["trial_design"], "trial", {"trial", "id", "reader"})
    ordered_trials = tuple(sorted(trials))
    for row in (*readers.values(), *sentences.values()):
        if type(row["fold"]) is not int or not 0 <= row["fold"] < FOLDS:
            raise ValueError("Invalid fold in the sealed trial design")
    if any(type(row["language"]) is not int or row["language"] not in PROFILE_CODES.values()
           for row in readers.values()):
        raise ValueError("Unexpected pre-reading language group")
    group_folds = {}
    for row in sentences.values():
        if group_folds.setdefault(row["group"], row["fold"]) != row["fold"]:
            raise ValueError("Normalized sentence variants cross folds")
    distinct_pairs = {(row["reader"], row["id"]) for row in trials.values()}
    if (len(readers) != READERS or len(sentences) != SENTENCES or len(group_folds) != SENTENCE_GROUPS
            or len(trials) != TRIALS or len(distinct_pairs) != TRIALS
            or {reader for reader, _ in distinct_pairs} != readers.keys()
            or {case for _, case in distinct_pairs} != sentences.keys()):
        raise ValueError("Unexpected trial, reader, or material inventory")
    reader_index = {reader: index for index, reader in enumerate(sorted(readers))}
    group_index = {group: index for index, group in enumerate(sorted(group_folds))}
    by_trial_reader = [readers[trials[trial]["reader"]] for trial in ordered_trials]
    by_trial_sentence = [sentences[trials[trial]["id"]] for trial in ordered_trials]
    reader_folds = np.array([row["fold"] for row in by_trial_reader])
    sentence_folds = np.array([row["fold"] for row in by_trial_sentence])
    language = np.array([row["language"] for row in by_trial_reader])
    cases = np.array([row["id"] for row in by_trial_sentence])
    counts = Counter(zip(reader_folds.tolist(), sentence_folds.tolist()))
    grid = np.array([[counts[i, j] for j in range(FOLDS)] for i in range(FOLDS)])
    if grid.tolist() != receipt["crossed_four_fold"]["test_counts"]:
        raise ValueError("Test-fold counts differ from the sealed design")
    fit_profiles, fit_stimuli = {}, {}
    for i, j in product(range(FOLDS), repeat=2):
        fit = (reader_folds != i) & (sentence_folds != j)
        fit_profiles[i, j] = np.bincount(language[fit], minlength=len(PROFILE_CODES)).tolist()
        fit_stimuli[i, j] = len(set(cases[fit]))
    return TrialDesign(
        ordered_trials, cases, np.array([group_index[row["group"]] for row in by_trial_sentence]),
        np.array([reader_index[row["reader"]] for row in by_trial_reader]), language, grid,
        fit_profiles, fit_stimuli,
    )


def verify_sources(args, plan):
    if read_json(args.score_protocol) != plan:
        raise ValueError("Scoring code or protocol differs from the frozen scoring receipt")
    prep = read_json(args.preparation)
    if (digest(args.preparation) != plan["preparation_receipt_sha256"] or prep["language_code"] != PROFILE_CODES
            or prep["protocol_sha256"] != plan["design_protocol_sha256"]):
        raise ValueError("The preparation receipt differs from the frozen design")
    paths = {
        "preparation_receipt": args.preparation,
        **{name: args.local / file for name, file in LOCAL_FILES.items()},
        **{name: args.extraction / file for name, file in FEATURE_FILES.items()},
    }
    bound = {key: digest(path) for key, path in paths.items()}
    for name, file in LOCAL_FILES.items():
        if prep["sha256"][f"local/{file}"] != bound[name]:
            raise ValueError(f"Unexpected local artifact {file}")
    seal = read_json(args.seal)
    if (seal["status"] != "fit_and_forecasts_locked_before_scoring"
            or seal["input_sha256"] != bound
            or seal["predictions_sha256"] != digest(args.forecasts)
            or seal["protocol_sha256"] != plan["fit_protocol_sha256"]
            or digest(args.fit_protocol) != plan["fit_protocol_sha256"]
            or seal["design_protocol_sha256"] != plan["design_protocol_sha256"]
            or seal["fit_code_sha256"] != digest(FIT_CODE)
            or seal["forecast_trials"] != TRIALS or tuple(seal["grades"]) != GRADES
            or len(seal["methods"]) != len(METHODS) or set(seal["methods"]) != set(METHODS)):
        raise ValueError("Forecast bytes, fit code or source files differ from the fit seal")
    design = read_design(args.local, prep)
    folds = seal["folds"]
    if len(folds) != FOLDS**2:
        raise ValueError("The fit seal omits a double-held-out test fold")
    for (i, j), fold in zip(product(range(FOLDS), repeat=2), folds, strict=True):
        scale, center = np.array(fold["length_scale"]), np.array(fold["length_center"])
        expected = (i, j, design.fold_grid[i, j], prep["crossed_four_fold"]["fit_counts"][i][j],
                    design.fit_stimulus_counts[i, j], design.fit_profile_counts[i, j])
        actual = tuple(fold[key] for key in ("reader_fold", "sentence_fold", "test_trials", "fit_trials",
                                              "fit_stimuli", "group_fit_counts"))
        if (actual != expected or scale.shape != (2,) or center.shape != (2,)
                or np.any(scale <= 0) or not np.isfinite(scale).all() or not np.isfinite(center).all()):
            raise ValueError(f"The frozen fit metadata differs from fold {(i, j)}")
    return design


def read_scores(outcomes_file, forecasts_file, design):
    forecasts = records(forecasts_file, "trial", {"trial", *METHODS})
    outcomes = records(outcomes_file, "trial", {"trial", "grade"})
    if forecasts.keys() != set(design.trials) or outcomes.keys() != forecasts.keys():
        raise ValueError("Every released trial must have one issued forecast and one grade")
    grades = [outcomes[key]["grade"] for key in design.trials]
    if any(type(grade) is not int or grade not in GRADES for grade in grades):
        raise ValueError("Unexpected ordinal outcome grade")
    probabilities = np.array([[forecasts[key][method] for method in METHODS] for key in design.trials])
    if (probabilities.shape != (TRIALS, len(METHODS), len(GRADES))
            or not np.all(np.isfinite(probabilities) & (probabilities > 0) & (probabilities <= 1))
            or not np.allclose(probabilities.sum(axis=2), 1, atol=1e-7, rtol=0)):
        raise ValueError("Expected all five finite, strictly positive grade probabilities")
    grades = np.array(grades)
    nll = -np.log(np.take_along_axis(probabilities, (grades - 1)[:, None, None], axis=2)[:, :, 0])
    cdf = np.cumsum(probabilities, axis=2)[:, :, :-1]
    rps = np.mean((cdf - (grades[:, None, None] <= np.array(GRADES[:-1]))) ** 2, axis=2)
    return np.stack((nll, rps), axis=-1), np.column_stack((grades, probabilities @ GRADES))


def effect_design(design):
    german, other = PROFILE_CODES["german"], PROFILE_CODES["another_language"]
    cases = np.intersect1d(design.literal_sentences[design.language == german],
                           design.literal_sentences[design.language == other])
    include = np.isin(design.literal_sentences, cases) & np.isin(design.language, (german, other))
    indices = np.flatnonzero(include)
    group_by_case = dict(zip(design.literal_sentences, design.normalized, strict=True))
    if len(cases) != PAIRED_SENTENCES:
        raise ValueError("The primary effect requires the 179 fixed literal sentence pairs")
    return EffectDesign(cases, np.array([group_by_case[case] for case in cases]), indices,
                        design.readers[indices], 2 * np.searchsorted(cases, design.literal_sentences[indices])
                        + (design.language[indices] == other))


def sentence_differences(reader_weights, measurements, reader_index, cells, cases):
    # This stage weights readers within each literal sentence and language cell.
    weight = reader_weights[:, reader_index]
    grouped = np.zeros((len(reader_weights), cases * 2, measurements.shape[1] + 1))
    indices = (np.arange(len(reader_weights))[:, None], cells[None, :])
    np.add.at(grouped[:, :, 0], indices, weight)
    np.add.at(grouped[:, :, 1:], indices, weight[:, :, None] * measurements[None, :, :])
    if np.any(grouped[:, :, 0] <= 0):
        raise ValueError("Every literal sentence and language cell needs an observed positive reader weight")
    mean = grouped[:, :, 1:] / grouped[:, :, :1]
    return mean[:, 1::2, :] - mean[:, 0::2, :]


def direction_agreement(differences):
    sign = np.sign(np.where(np.abs(differences) <= TIE_ATOL, 0, differences))
    return (1 + sign[..., :1] * sign[..., 1:]) / 2


def resample(design, effect, losses, values):
    reader_seed, group_seed = np.random.SeedSequence(SEED).spawn(2)
    reader_rng, group_rng = np.random.default_rng(reader_seed), np.random.default_rng(group_seed)
    metrics = np.empty((DRAWS, len(METHODS), len(METRICS)))
    contrasts = np.empty((DRAWS, len(METHODS) + 1))
    directions = np.empty((DRAWS, len(METHODS)))
    for start in range(0, DRAWS, BATCH):
        block = slice(start, min(start + BATCH, DRAWS))
        count = block.stop - block.start
        readers = reader_rng.exponential(size=(count, READERS))
        groups = group_rng.exponential(size=(count, SENTENCE_GROUPS))
        trial_weight = readers[:, design.readers] * groups[:, design.normalized]
        metrics[block] = (trial_weight @ losses.reshape(TRIALS, -1) / trial_weight.sum(axis=1, keepdims=True)
                          ).reshape(count, len(METHODS), len(METRICS))
        literal_weight = groups[:, effect.normalized]
        literal_weight /= literal_weight.sum(axis=1, keepdims=True)
        difference = sentence_differences(readers, values[effect.trial_index], effect.readers, effect.cells, len(effect.cases))
        contrasts[block] = np.einsum("nf,nfc->nc", literal_weight, difference)
        directions[block] = np.einsum("nf,nfc->nc", literal_weight, direction_agreement(difference))
    return metrics, contrasts, directions


def interval(value, draws, confidence=95):
    alpha = (100 - confidence) / 200
    return {"estimate": float(value), f"ci{confidence}": np.quantile(draws, [alpha, 1 - alpha]).tolist()}


def case_reports(effect, values, point):
    # Replay the same reader weights in small sentence blocks instead of retaining all draws.
    reader_seed = np.random.SeedSequence(SEED).spawn(2)[0]
    for start in range(0, len(effect.cases), SENTENCE_BATCH):
        end = min(start + SENTENCE_BATCH, len(effect.cases))
        reader_rng = np.random.default_rng(reader_seed)
        select = (effect.cells >= start * 2) & (effect.cells < end * 2)
        measurements = values[effect.trial_index[select]]
        draws = np.empty((DRAWS, end - start, len(METHODS) + 1))
        for offset in range(0, DRAWS, BATCH):
            block = slice(offset, min(offset + BATCH, DRAWS))
            readers = reader_rng.exponential(size=(block.stop - block.start, READERS))
            draws[block] = sentence_differences(
                readers, measurements, effect.readers[select], effect.cells[select] - 2 * start, end - start,
            )
        for j in range(start, end):
            sample = draws[:, j - start, :]
            yield {
                "id": str(effect.cases[j]), "human": interval(point[j, 0], sample[:, 0]),
                "models": {method: {
                    "difference": interval(point[j, i], sample[:, i]),
                    "minus_human": interval(point[j, i] - point[j, 0], sample[:, i] - sample[:, 0]),
                } for i, method in enumerate(METHODS, 1)},
            }


def summarize(design, losses, values):
    effect = effect_design(design)
    literal = sentence_differences(np.ones((1, READERS)), values[effect.trial_index], effect.readers,
                                   effect.cells, len(effect.cases))[0]
    measured, *modeled = literal.mean(axis=0)
    losses_mean = losses.mean(axis=0)
    resampled_losses, deltas, directions = resample(design, effect, losses, values)
    comparisons = {}
    for first, second in COMPARISONS:
        i, j = METHODS.index(first), METHODS.index(second)
        comparisons[f"{first}_minus_{second}"] = {
            metric: interval(losses_mean[i, k] - losses_mean[j, k], resampled_losses[:, i, k] - resampled_losses[:, j, k])
            for k, metric in enumerate(METRICS)
        }
    human = interval(measured, deltas[:, 0])
    effects = {
        method: {
            "group_difference": interval(modeled[i], deltas[:, i + 1]),
            "group_difference_minus_human": interval(modeled[i] - measured, deltas[:, i + 1] - deltas[:, 0], 90),
            "per_sentence_direction_agreement": interval(direction_agreement(literal).mean(axis=0)[i], directions[:, i]),
        } for i, method in enumerate(METHODS)
    }
    candidate = effects["qwen_interaction"]
    lower, upper = human["ci95"]
    eq_lower, eq_upper = candidate["group_difference_minus_human"]["ci90"]
    conditions = {
        "matched_nll_ci95_upper_below_zero": comparisons["qwen_interaction_minus_qwen_additive"]["nll_nats"]["ci95"][1] < 0,
        "human_ci95_excludes_zero": lower > 0 or upper < 0,
        "candidate_direction_matches_human": candidate["group_difference"]["estimate"] * human["estimate"] > 0,
        "candidate_minus_human_ci90_within_half_grade": -EQUIVALENCE < eq_lower and eq_upper < EQUIVALENCE,
    }
    return {
        "trial_metrics": {method: {
            metric: interval(losses_mean[i, k], resampled_losses[:, i, k]) for k, metric in enumerate(METRICS)
        } for i, method in enumerate(METHODS)},
        "matched_trial_comparisons": comparisons,
        "same_literal_sentence_effect": {
            "human": human, "methods": effects, "literal_sentences": len(effect.cases),
            "normalized_sentence_groups": len(set(effect.normalized)), "observations": len(effect.trial_index),
            "by_sentence": list(case_reports(effect, values, literal)),
        },
        "interaction_candidate_gate": {**conditions, "passes": all(conditions.values())},
    }


def score_protocol():
    return {
        "status": "scoring_rules_locked_before_real_scoring",
        "design_protocol_sha256": digest(DESIGN_PROTOCOL),
        "preparation_receipt_sha256": PREPARATION_SHA256,
        "fit_protocol_sha256": FIT_PROTOCOL_SHA256,
        "scorer_sha256": digest(Path(__file__)),
        "counts": {"readers": READERS, "sentences": SENTENCES, "normalized_sentences": SENTENCE_GROUPS,
                   "trials": TRIALS, "test_folds": FOLDS**2, "same_literal_sentence_pairs": PAIRED_SENTENCES},
        "methods": list(METHODS),
        "grades": list(GRADES),
        "matched_comparisons": [[first, second] for first, second in COMPARISONS],
        "trial_primary": "qwen_interaction minus qwen_additive observed grade NLL, nats",
        "trial_secondary": "mean squared error of four cumulative ordinal probabilities",
        "effect_primary": "other-language minus German-only after averaging observed and predicted grades in each literal "
                          "sentence by group, all eligible literal sentences equal weight",
        "secondary_effect": "mean same-sentence sign agreement across the fixed pairs; half credit for a tie",
        "sign_tie_atol": TIE_ATOL,
        "intervals": "paired percentile 95%, except model minus human equivalence interval 90%",
        "candidate_gate": "NLL difference 95% upper bound <0, human effect 95% CI excludes 0, matching effect direction, "
                          "90% model minus human CI strictly inside the equivalence margin",
        "equivalence_margin_grades": EQUIVALENCE,
        "bootstrap": {"draws": DRAWS, "seed": SEED, "block": BATCH, "per_sentence_block": SENTENCE_BATCH,
                      "reader_weight": "independent exponential mean 1",
                      "normalized_sentence_weight": "independent exponential mean 1",
                      "frozen_oof_predictions": True,
                      "proper_trial_score": "reader weight times normalized sentence weight, then normalize",
                      "average_effect": "reader weight within each literal sentence group mean; normalized sentence weight "
                                        "across literal sentence effects",
                      "individual_sentence_effect": "reader weights; its shared sentence weight cancels"},
    }


def write_json(path, data):
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")


def score(args):
    plan = score_protocol()
    design = verify_sources(args, plan)
    losses, values = read_scores(args.local / LOCAL_FILES["outcomes"], args.forecasts, design)
    metrics = summarize(design, losses, values)
    write_json(args.output, {
        "status": "scored_locally_against_original_human_responses",
        "score_protocol_sha256": digest(args.score_protocol),
        "fit_seal_sha256": digest(args.seal),
        "forecasts_sha256": digest(args.forecasts),
        "counts": plan["counts"],
        **metrics,
    })


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    subcommands = cli.add_subparsers(dest="action", required=True)
    freeze = subcommands.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = subcommands.add_parser("score")
    for name in ("preparation", "local", "extraction", "fit-protocol", "score-protocol", "forecasts", "seal", "output"):
        evaluate.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    if args.action == "freeze":
        write_json(args.output, score_protocol())
    else:
        score(args)


if __name__ == "__main__":
    main()
