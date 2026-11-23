"""Seal frozen ACT forecasts and score first-correct positions in a separate step."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logsumexp

from research.semantic_tracking.beukers_extract import FP16_TOLERANCE, read_stimuli, validate_preflight
from research.semantic_tracking.core_vocabulary_disjoint_score import (
    ARCHIVED_SHA256, frozen_feature_blocks, read_json,
)
from research.semantic_tracking.core_vocabulary_marginal_fit import (
    BOOTSTRAPS, BOOTSTRAP_SEED, CDF_MARGIN, METHODS, PROJECTION_WIDTH, TYPE, interval, surface_features,
)
from research.semantic_tracking.core_vocabulary_marginal_score import read_jsonl, sha256, write_json


PROTOCOL_SHA256 = "fec8687fde463af8baf19da310b951c51238ea5fd60c5475f56bfdbc43ed6010"
MATERIAL_SHA256 = "f42050df403a163aeff0c5468443cd4edf0e738bdfa2fbb48fb3cd6aa45dd5ab"
SPLIT_SHA256 = "f7492e652e82608c3c59ef83e589e91b7d05e734cde5e892bae2ae42a93920c7"
RUNTIME_PREFLIGHT_SHA256 = "cad4e7293e348c6c7e703b413b274c593d527ed34896423cc8cce01624a6b19c"
ASSESSMENT_SHA256 = "5df3dcf921a808b32cb38bb762a7b45dab343271ec0981b1b7a64817f2376dae"
OUTCOMES_SHA256 = "98fdf38a61b094d31b4b73ef5d43b2641d1241a8fd4903720947419c63ef7af4"
DOCUMENTS, READERS, FOLDS = 20, 100, 5
PRIMARY_HORIZON, FULL_HORIZON = 6, 15
PRIMARY_STEPS, FULL_STEPS = (1, 3, 6), (3, 6, 10, 15)
STIMULI_FILE = "model_input/stimuli.jsonl"
FORECAST_FILE, SEAL_FILE = "forecasts.jsonl", "seal.json"
IDENTIFIER = re.compile(r"[0-9a-f]{32}")


@dataclass(frozen=True)
class Inputs:
    stimuli: list[dict]
    families: list[dict]
    binding: dict


def checked(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Unexpected frozen input SHA-256: {path.name}")


def verify_inputs(material: Path, runtime_preflight: Path, archived: Path, protocol: Path, extraction: Path) -> Inputs:
    pins = {
        material / "preflight.json": MATERIAL_SHA256, material / "split.json": SPLIT_SHA256,
        runtime_preflight: RUNTIME_PREFLIGHT_SHA256, protocol: PROTOCOL_SHA256,
        **{archived / name: digest for name, digest in ARCHIVED_SHA256.items()},
    }
    for path, digest in pins.items():
        checked(path, digest)
    for name, digest in read_json(material / "preflight.json")["file_sha256"].items():
        checked(material / name, digest)

    stimuli_file = material / STIMULI_FILE
    stimuli = read_stimuli(stimuli_file)
    manifest = read_json(runtime_preflight)
    validate_preflight(manifest, stimuli, sha256(stimuli_file))
    families = read_json(material / "split.json")["material"]
    if (len(stimuli) != DOCUMENTS or len(families) != DOCUMENTS
            or len({row["family"] for row in families}) != DOCUMENTS
            or Counter(row["fold"] for row in families) != {i: DOCUMENTS // FOLDS for i in range(FOLDS)}):
        raise ValueError("Expected the frozen 20 ACT lists and five balanced folds")
    for stimulus, family in zip(stimuli, families, strict=True):
        if (tuple(stimulus) != ("id", "stops", "text") or tuple(family) != ("family", "fold", "id")
                or stimulus["id"] != family["id"] or len(stimulus["stops"]) != FULL_HORIZON
                or stimulus["stops"][-1] != len(stimulus["text"])):
            raise ValueError("Unexpected, reordered or leaking ACT model inputs")

    original = read_json(archived / "extraction-seal.json")
    seal = read_json(extraction / SEAL_FILE)
    lineage = ("candidate_id", "model", "revision", "tokenizer_sha256", "transformers", "anchor_heads_sha256",
               "projection_sha256", "projection_width", "source_code_sha256", "gpu", "compute_dtype", "storage_dtype")
    if any(seal.get(key) != original[key] for key in lineage):
        raise ValueError("ACT extraction differs from the frozen model, tokenizer or projection")
    if (any(seal.get(key) != value for key, value in manifest.items())
            or seal.get("assay_sha256") != MATERIAL_SHA256 or seal.get("preflight_sha256") != RUNTIME_PREFLIGHT_SHA256
            or seal.get("protocol_sha256") != PROTOCOL_SHA256 or seal.get("content_forward_per_document") != 1
            or seal.get("content_batches") != 1 or seal.get("vectors_shape") != [DOCUMENTS * FULL_HORIZON, PROJECTION_WIDTH]
            or seal.get("independent_causal_checks") != 12
            or not 0 <= seal.get("max_storage_relative_l2", math.inf) <= FP16_TOLERANCE):
        raise ValueError("Extraction diverges from the projection-only ACT protocol")
    names = {"coordinates.jsonl", "vectors.npy", "causality.json", "telemetry.jsonl"}
    if seal["file_sha256"].keys() != names:
        raise ValueError("Extraction must contain only certified coordinates and the projected features")
    for name in names:
        checked(extraction / name, seal["file_sha256"][name])
    coordinates = read_jsonl(extraction / "coordinates.jsonl")
    for row, stimulus in zip(coordinates, stimuli, strict=True):
        if tuple(row) != ("id", "stops") or row != {"id": stimulus["id"], "stops": stimulus["stops"]}:
            raise ValueError("Extracted coordinate order differs from the source cue stops")
    checks = read_json(extraction / "causality.json")["checks"]
    comparisons = {"substring", "right_padding", "suffix_replacement", "changed_suffix_positive_control"}
    observed = Counter((row["id"], row["stop"]) for row in checks)
    expected = {(key, stop, method) for key, stop in observed for method in comparisons}
    if (len(observed) != 3 or len(checks) != 12
            or {(row["id"], row["stop"], row["comparison"]) for row in checks} != expected
            or any(row["passed"] is not True or row["projected_tokens"]["passed"] is not True for row in checks)):
        raise ValueError("ACT causal prefix controls did not all pass")
    binding = {
        "schema_version": 1, "protocol_sha256": PROTOCOL_SHA256, "split_sha256": SPLIT_SHA256,
        "material_preflight_sha256": MATERIAL_SHA256, "runtime_preflight_sha256": RUNTIME_PREFLIGHT_SHA256,
        "extraction_seal_sha256": sha256(extraction / SEAL_FILE), "original_artifact_sha256": ARCHIVED_SHA256,
        "scorer_sha256": sha256(Path(__file__)),
    }
    return Inputs(stimuli, families, binding)


def visible_surface(text: str, stops: list[int]) -> np.ndarray:
    cumulative = 0
    earlier_types = set()
    features = []
    start = 0
    for ordinal, stop in enumerate(stops, start=1):
        cue = text[start:stop]
        if (not cue or "\n" in cue or (ordinal < FULL_HORIZON and text[stop:stop + 1] != "\n")):
            raise ValueError("ACT stop must terminate the exact displayed cue")
        words = len(cue.split())
        cumulative += words
        cue_types = set(TYPE.findall(unicodedata.normalize("NFKC", cue).casefold()))
        features.append((math.log1p(words), math.log1p(cumulative), len(cue_types & earlier_types) / max(1, len(cue_types))))
        earlier_types.update(cue_types)
        start = stop + 1
    surface = np.asarray(features, dtype=np.float64)
    cutoff = stops[PRIMARY_HORIZON - 1]
    if not np.array_equal(surface[:PRIMARY_HORIZON], surface_features(text[:cutoff], stops[:PRIMARY_HORIZON])):
        raise ValueError("ACT visible features drifted from the frozen English feature definitions")
    return surface


def log_masses(logits: np.ndarray) -> np.ndarray:
    if not np.isfinite(logits).all():
        raise ValueError("Expected finite conditional hazard logits")
    survive = np.cumsum(-np.logaddexp(0, logits), axis=-1)
    preceding = np.concatenate((np.zeros_like(survive[..., :1]), survive[..., :-1]), axis=-1)
    return np.concatenate((preceding - np.logaddexp(0, -logits), survive[..., -1:]), axis=-1)


def fixed_forecasts(stimuli: list[dict], vectors: np.ndarray, fit: dict) -> tuple[np.ndarray, np.ndarray]:
    if vectors.shape != (DOCUMENTS * FULL_HORIZON, PROJECTION_WIDTH) or vectors.dtype != np.float16:
        raise ValueError("Expected a stored FP16 feature row at all 300 ACT word stops")
    states = vectors.reshape(DOCUMENTS, FULL_HORIZON, PROJECTION_WIDTH)
    features = frozen_feature_blocks(fit, np.stack([visible_surface(row["text"], row["stops"]) for row in stimuli]), states)
    offsets = np.stack([np.zeros((DOCUMENTS, FULL_HORIZON)),
                        *[features[name] @ fit["models"][name]["slopes"] for name in METHODS[1:]]])
    position = np.broadcast_to(fit["position_log_masses"], (DOCUMENTS, PRIMARY_HORIZON + 1))
    primary = np.stack([position, *[
        log_masses(offsets[i, :, :PRIMARY_HORIZON] + fit["models"][name]["intercepts"])
        for i, name in enumerate(METHODS[1:], start=1)
    ]])
    if not np.isfinite(offsets).all() or not np.allclose(logsumexp(primary, axis=-1), 0, rtol=0, atol=1e-12):
        raise ValueError("Invalid frozen English ACT forecast")
    return primary, offsets


def stage_forecast(material, runtime_preflight, archived, protocol, extraction, output):
    if output.exists():
        raise FileExistsError(output)
    inputs = verify_inputs(material, runtime_preflight, archived, protocol, extraction)
    vectors = np.load(extraction / "vectors.npy", mmap_mode="r", allow_pickle=False)
    primary, offsets = fixed_forecasts(inputs.stimuli, vectors, read_json(archived / "fit.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".act-forecast-") as home:
        stage = Path(home) / "sealed"
        stage.mkdir()
        with (stage / FORECAST_FILE).open("x", encoding="utf-8", newline="\n") as stream:
            for index, stimulus in enumerate(inputs.stimuli):
                row = {
                    "id": stimulus["id"],
                    "fixed_six_log_masses": {name: primary[m, index].tolist() for m, name in enumerate(METHODS)},
                    "full_fifteen_logit_offsets": {name: offsets[m, index].tolist() for m, name in enumerate(METHODS)},
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(stage / SEAL_FILE, {**inputs.binding, "forecasts_sha256": sha256(stage / FORECAST_FILE)})
        stage.rename(output)


def read_forecasts(path: Path, inputs: Inputs) -> tuple[np.ndarray, np.ndarray]:
    if read_json(path / SEAL_FILE) != {**inputs.binding, "forecasts_sha256": sha256(path / FORECAST_FILE)}:
        raise ValueError("Grade-free ACT forecasts must be sealed before opening outcomes")
    fixed, offsets = [], []
    keys = ("id", "fixed_six_log_masses", "full_fifteen_logit_offsets")
    for row, original in zip(read_jsonl(path / FORECAST_FILE), inputs.stimuli, strict=True):
        if tuple(row) != keys or row["id"] != original["id"]:
            raise ValueError("Unexpected or reordered ACT forecast identities")
        for field, width, rows in ((keys[1], PRIMARY_HORIZON + 1, fixed), (keys[2], FULL_HORIZON, offsets)):
            methods = row[field]
            if tuple(methods) != METHODS or any(len(values) != width or any(
                type(x) not in (int, float) or not math.isfinite(x) for x in values
            ) for values in methods.values()):
                raise ValueError("Unexpected ACT method, forecast width or numerical value")
            rows.append(list(methods.values()))
    fixed, offsets = (np.asarray(rows, dtype=np.float64).transpose(1, 0, 2) for rows in (fixed, offsets))
    if (not np.allclose(logsumexp(fixed, axis=-1), 0, rtol=0, atol=1e-12) or offsets[0].any()
            or not np.all(fixed[0] == fixed[0, 0])):
        raise ValueError("Invalid fixed primary distribution or nonzero position-only offset")
    return fixed, offsets


def observations(assessment: Path, families: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    checked(assessment / "assessment.json", ASSESSMENT_SHA256)
    checked(assessment / "outcomes.jsonl", OUTCOMES_SHA256)
    receipt = read_json(assessment / "assessment.json")
    if (receipt["material_preflight_sha256"] != MATERIAL_SHA256 or receipt["split_sha256"] != SPLIT_SHA256
            or receipt["outcomes_sha256"] != OUTCOMES_SHA256):
        raise ValueError("Historical outcomes disagree with the frozen ACT family coordinates")
    rows = read_jsonl(assessment / "outcomes.jsonl")
    readers = {key: i for i, key in enumerate(sorted({row["reader_id"] for row in rows}))}
    ids = {row["family"]: i for i, row in enumerate(families)}
    if len(readers) != READERS or len(rows) != DOCUMENTS * READERS:
        raise ValueError("Unexpected ACT reader-by-list outcome grid")
    primary = np.zeros((READERS, DOCUMENTS, PRIMARY_HORIZON + 1), dtype=np.float64)
    full = np.zeros((READERS, DOCUMENTS, FULL_HORIZON + 1), dtype=np.float64)
    seen = set()
    unknown = 0
    fields = ("family", "first_correct", "original_first_correct", "original_solved", "reader_id", "status")
    for row in rows:
        if (tuple(row) != fields or row["family"] not in ids or not IDENTIFIER.fullmatch(row["reader_id"])
                or type(row["original_solved"]) is not bool):
            raise ValueError("Unexpected pseudonymized ACT outcome schema")
        index = (readers[row["reader_id"]], ids[row["family"]])
        solved, event, status = (row[key] for key in ("original_solved", "original_first_correct", "status"))
        if index in seen or (event is not None and (type(event) is not int or not 1 <= event <= FULL_HORIZON or not solved)):
            raise ValueError("Repeated or invalid first-correct position")
        seen.add(index)
        is_unknown = solved and event is None
        early = event if event is not None and event <= PRIMARY_HORIZON else None
        expected = "unknown" if is_unknown else "first_correct" if early is not None else "right_censored"
        if status != expected or row["first_correct"] != early:
            raise ValueError("Released primary and full ACT endpoints disagree")
        if is_unknown:
            unknown += 1
            continue
        primary[(*index, early - 1 if early is not None else PRIMARY_HORIZON)] = 1
        full[(*index, event - 1 if event is not None else FULL_HORIZON)] = 1
    totals_match = np.array_equal(primary.sum(axis=(0, 1))[:-1], receipt["first_correct"]) and np.array_equal(
        full.sum(axis=(0, 1))[:-1], receipt["full_first_correct"]
    )
    if unknown != 1 or primary.sum() != 1999 or full.sum() != 1999 or not totals_match:
        raise ValueError("ACT historical response inventory did not reconcile with its release")
    return primary, full


def equal_list_intercept(offset: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    active = risk > 0
    if not active.any():
        return 0.0  # The balanced pseudo-trial alone has intercept zero.
    offsets = offset[active]
    event_proportion = events[active] / risk[active]
    mean_risk = float(risk[active].mean())

    def derivative(intercept):
        return float(mean_risk * (expit(offsets + intercept) - event_proportion).sum() + expit(intercept) - 0.5)

    bound = float(np.max(np.abs(offsets))) + math.log1p(2 * risk.sum()) + 2
    return float(brentq(derivative, -bound, bound, xtol=1e-13))


def fit_full_curve(offsets: np.ndarray, outcomes: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, dict]:
    totals = outcomes.sum(axis=0)
    events = totals[:, :FULL_HORIZON]
    prior_events = np.concatenate((np.zeros_like(events[:, :1]), np.cumsum(events[:, :-1], axis=-1)), axis=-1)
    risk = totals.sum(axis=-1, keepdims=True) - prior_events
    logits = np.full_like(offsets, np.nan)
    fitted = {name: {} for name in METHODS}
    for fold in range(FOLDS):
        train = folds != fold
        for index, name in enumerate(METHODS):
            intercepts = [equal_list_intercept(offsets[index, train, ordinal], events[train, ordinal], risk[train, ordinal])
                          for ordinal in range(FULL_HORIZON)]
            logits[index, ~train] = offsets[index, ~train] + intercepts
            fitted[name][str(fold)] = intercepts
    return log_masses(logits), fitted


def evaluate(masses: np.ndarray, outcomes: np.ndarray, steps: tuple[int, ...], draws: int = BOOTSTRAPS,
             *, primary_gates: bool = True) -> dict:
    if (masses.shape != (len(METHODS), *outcomes.shape[1:]) or not np.isfinite(masses).all()
            or not np.allclose(logsumexp(masses, axis=-1), 0, rtol=0, atol=1e-12)):
        raise ValueError("Expected finite, normalized ACT outcome masses for all three methods")
    n_readers, n_lists, _ = outcomes.shape
    truth_counts = outcomes.sum(axis=0)
    truth = truth_counts / truth_counts.sum(axis=-1, keepdims=True)
    stage_indices = np.asarray(steps) - 1
    truth_cdf = truth.cumsum(axis=-1)[..., stage_indices]
    model_cdf = np.exp(masses).cumsum(axis=-1)[..., stage_indices]
    loss = -np.einsum("fk,mfk->mf", truth, masses)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(n_lists, size=(draws, n_lists))
    list_losses = loss[:, sampled].mean(axis=-1)
    list_human_cdf = truth_cdf[sampled].mean(axis=1)
    list_model_cdf = model_cdf[:, sampled].mean(axis=-2)
    crossed_human_cdf = np.empty_like(list_human_cdf)
    crossed_losses = np.empty_like(list_losses)
    readers = np.full(n_readers, 1 / n_readers)
    for start in range(0, draws, 128):
        stop = min(start + 128, draws)
        weights = rng.multinomial(n_readers, readers, size=stop - start)
        counts = (weights @ outcomes.reshape(n_readers, -1)).reshape(-1, *truth.shape)
        chosen = sampled[start:stop]
        counts = counts[np.arange(stop - start)[:, None], chosen]
        at_risk = counts.sum(axis=-1, keepdims=True)
        if (at_risk == 0).any():
            raise ValueError("Crossed bootstrap lost all responses for a selected ACT list")
        weights_at_list = counts / at_risk
        crossed_human_cdf[start:stop] = weights_at_list.cumsum(axis=-1)[..., stage_indices].mean(axis=1)
        crossed_losses[:, start:stop] = -np.einsum("dfk,mdfk->md", weights_at_list, masses[:, chosen]) / n_lists

    superiority = {}
    for index, name in enumerate(METHODS[:-1]):
        limit = interval(list_losses[index] - list_losses[-1])
        point = float((loss[index] - loss[-1]).mean())
        superiority[name] = {
            "control_minus_qwen_nll_nats": point, "paired_list_95_ci": limit,
            "paired_crossed_95_ci": interval(crossed_losses[index] - crossed_losses[-1]),
        }
        if primary_gates:
            superiority[name]["passes"] = point > 0 and limit[0] > 0
    calibration = {}
    for i, name in enumerate(METHODS):
        calibration[name] = []
        for j, step in enumerate(steps):
            limit = interval(list_model_cdf[i, :, j] - list_human_cdf[:, j])
            row = {
                "exposure": step, "human_cdf": float(truth_cdf[:, j].mean()),
                "model_cdf": float(model_cdf[i, :, j].mean()),
                "signed_model_minus_human": float((model_cdf[i, :, j] - truth_cdf[:, j]).mean()),
                "paired_list_95_ci": limit,
                "paired_crossed_95_ci": interval(list_model_cdf[i, :, j] - crossed_human_cdf[:, j]),
            }
            if primary_gates:
                row["within_margin"] = limit[0] > -CDF_MARGIN and limit[1] < CDF_MARGIN
            calibration[name].append(row)
    result = {
        "list_macro_nll_nats": {name: float(loss[i].mean()) for i, name in enumerate(METHODS)},
        "paired_nll_superiority": superiority, "cdf_calibration": calibration,
        "bootstrap": {"draws": draws, "seed": BOOTSTRAP_SEED, "interval": 0.95, "primary_clusters": "word_list",
                      "crossed_clusters": ["word_list", "reader"]},
        "observed_list_macro_outcome_probabilities": truth.mean(axis=0).tolist(),
    }
    if primary_gates:
        result.update(passes_nll=all(row["passes"] for row in superiority.values()), cdf_margin=CDF_MARGIN,
                      passes_qwen_cdf=all(row["within_margin"] for row in calibration["qwen"]))
    return result


def stage_score(material, runtime_preflight, archived, protocol, extraction, output, forecasts, assessment):
    if output.exists():
        raise FileExistsError(output)
    inputs = verify_inputs(material, runtime_preflight, archived, protocol, extraction)
    fixed, offsets = read_forecasts(forecasts, inputs)
    primary, full = observations(assessment, inputs.families)
    folds = np.array([row["fold"] for row in inputs.families])
    crossfit, intercepts = fit_full_curve(offsets, full, folds)
    primary_result = evaluate(fixed, primary, PRIMARY_STEPS)
    secondary = evaluate(crossfit, full, FULL_STEPS, primary_gates=False)
    secondary["intercept_fit"] = {
        "folds": FOLDS,
        "nonempty_training_risk_sets": "Equal list weight, with each list scaled to the step's mean at-risk count",
        "pseudo_trial": {"offset": 0.0, "event_weight": 0.5, "nonevent_weight": 0.5},
        "bootstrap_refits_intercepts": False,
    }
    secondary["fitted_fold_intercepts"] = intercepts
    secondary["list_fold_predictions"] = [
        {"id": row["id"], "fold": row["fold"],
         "log_masses": {name: crossfit[i, j].tolist() for i, name in enumerate(METHODS)}}
        for j, row in enumerate(inputs.families)
    ]
    write_json(output, {"forecast_seal_sha256": sha256(forecasts / SEAL_FILE),
                        "assessment_sha256": ASSESSMENT_SHA256, "outcomes_sha256": OUTCOMES_SHA256,
                        "known_trials": 1999, "unknown_trials": 1, "word_lists": DOCUMENTS, "readers": READERS,
                        "primary_fixed_six": primary_result, "secondary_crossfit_fifteen": secondary})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="command", required=True)
    for name in ("forecast", "score"):
        command = stages.add_parser(name)
        for field in ("material", "runtime-preflight", "archived", "protocol", "extraction", "output"):
            command.add_argument(f"--{field}", type=Path, required=True)
        if name == "score":
            command.add_argument("--forecasts", type=Path, required=True)
            command.add_argument("--assessment", type=Path, required=True)
    args = vars(parser.parse_args())
    (stage_forecast if args.pop("command") == "forecast" else stage_score)(**args)


if __name__ == "__main__":
    main()
