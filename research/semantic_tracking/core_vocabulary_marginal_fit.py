"""Fit causal Core Vocabulary readouts and open held-out outcomes after sealing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from research.semantic_tracking.core_vocabulary_marginal_prepare import ORDINALS, SEPARATOR, target_family
from research.semantic_tracking.core_vocabulary_marginal_score import (
    CONDITIONS, TRIAL_TABLE_SHA256, aggregate_trials, forecast_mixtures, inventory,
    position_log_masses, read_jsonl, read_packet, read_split, sha256, write_json,
)
from research.semantic_tracking.gum_bridge_extract import CANDIDATE, PROJECTION_WIDTH, TOKENIZER_SHA256


MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
ANCHOR_SHA256 = "c7f45af167c320e8f7065a3fb9a45d59aa4c2ff912f6f415f93500c028a90f18"
ASSAY_SHA256 = "4fa6258798bc3f76410276491b985031f4c7e8a0d156063fd0908b1bb4451bdd"
SPLIT_SHA256 = "2af0ba7e968d2fa3acfb2b6f774a09cd68819986232b82d5273724af36f3894b"
PROTOCOL_SHA256 = "8acc588ee962a9749d85a7c7fed4c0fbe9db38bdf02a968e9262dc7871e59f6f"
EXTRACTOR_PREFLIGHT_SHA256 = "e0ac2654b530b64e94b2d0cedcc12b7c692472b9e5074e570b5b3f09f7bc3537"
METHODS = ("position", "surface", "qwen")
SURFACE = ("log1p_current_words", "log1p_cumulative_words", "prior_type_overlap")
TYPE = re.compile(r"[^\W_]+")
RIDGE = 0.10
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 735972
CDF_STEPS = (1, 3, 6)
CDF_MARGIN = 0.05


def surface_features(text: str, stops: list[int]) -> np.ndarray:
    if len(stops) != ORDINALS or stops[-1] != len(text):
        raise ValueError("Expected six exact clue ends")
    start = 0
    words_seen = 0
    old_types = set()
    features = []
    for index, stop in enumerate(stops):
        clue = text[start:stop]
        if not clue or SEPARATOR in clue or (index < ORDINALS - 1 and text[stop:stop + 1] != SEPARATOR):
            raise ValueError("Cue stop does not preserve the source line")
        words = len(clue.split())
        words_seen += words
        normalized = unicodedata.normalize("NFKC", clue).casefold()
        types = set(TYPE.findall(normalized))
        features.append((math.log1p(words), math.log1p(words_seen), len(types & old_types) / max(1, len(types))))
        old_types.update(types)
        start = stop + len(SEPARATOR)
    return np.asarray(features, dtype=np.float64)


def objective(theta: np.ndarray, data: np.ndarray, target_prob: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the target-macro mixture NLL plus ridge and its analytic gradient."""
    slopes = theta[ORDINALS:]
    logits = data @ slopes + theta[:ORDINALS]
    log_survive = -np.logaddexp(0, logits)
    log_hazard = -np.logaddexp(0, -logits)
    cumulative = np.cumsum(log_survive, axis=-1)
    previous = np.concatenate((np.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), axis=-1)
    path = np.concatenate((previous + log_hazard, cumulative[..., -1:]), axis=-1)
    summed = logsumexp(path, axis=1)
    log_marginal = summed - math.log(logits.shape[1])
    outcome_weights = target_prob / len(data)
    loss = -np.sum(outcome_weights * log_marginal) + RIDGE / 2 * (slopes @ slopes)
    responsibility = np.exp(path - summed[:, None, :]) * outcome_weights[:, None, :]
    remaining = np.cumsum(responsibility[..., ::-1], axis=-1)[..., :0:-1]
    derivative = expit(logits) * remaining - responsibility[..., :-1]
    gradient = np.concatenate((
        derivative.sum(axis=(0, 1)),
        data.reshape(-1, data.shape[-1]).T @ derivative.ravel() + RIDGE * slopes,
    ))
    return float(loss), gradient


def fit_readout(data: np.ndarray, target_prob: np.ndarray, initial_stage: np.ndarray) -> tuple[np.ndarray, dict]:
    initial = np.concatenate((initial_stage, np.zeros(data.shape[-1])))
    fit = minimize(
        objective, initial, args=(data, target_prob), jac=True, method="L-BFGS-B",
        options={"maxiter": 750, "ftol": 1e-13, "gtol": 1e-9},
    )
    if not fit.success or not np.isfinite(fit.x).all() or not np.isfinite(fit.fun):
        raise RuntimeError(f"Core Vocabulary mixture optimization failed: {fit.message}")
    return fit.x, {
        "iterations": int(fit.nit), "penalized_train_loss": float(fit.fun),
        "max_absolute_gradient": float(np.max(np.abs(fit.jac))),
        "intercepts": fit.x[:ORDINALS].tolist(), "slopes": fit.x[ORDINALS:].tolist(),
    }


def standardize(surface: np.ndarray, states: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    scalar_rows = surface[train].reshape(-1, surface.shape[-1])
    mean = scalar_rows.mean(axis=0)
    std = scalar_rows.std(axis=0)
    std[std == 0] = 1
    controls = (surface - mean) / std

    norms = np.linalg.norm(states, axis=-1, keepdims=True)
    if not np.allclose(norms, 1, rtol=0, atol=0.002):
        raise ValueError("Expected finite unit Qwen vectors within float16 tolerance")
    unit = states / norms
    vector_rows = unit[train].reshape(-1, unit.shape[-1])
    center = vector_rows.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum((vector_rows - center) ** 2, axis=-1))))
    if not math.isfinite(rms) or rms <= 0:
        raise ValueError("No Qwen state variation in training")
    combined = np.concatenate((controls, (unit - center) / rms), axis=-1)
    return controls, combined, {
        "surface_features": list(SURFACE), "surface_means": mean.tolist(), "surface_scales": std.tolist(),
        "qwen_unit_vector_mean": center.tolist(), "qwen_centered_rms_l2": rms,
    }


def extraction_inputs(packet: Path, run: Path, protocol: Path, extractor_preflight: Path) -> tuple[list, np.ndarray]:
    stimuli_file = packet / "model_input" / "stimuli.jsonl"
    seal = json.loads((run / "seal.json").read_text(encoding="utf-8"))
    stimuli = read_jsonl(stimuli_file)
    expected = {
        "candidate_id": CANDIDATE, "revision": MODEL_REVISION, "model": "Qwen/Qwen3.5-2B",
        "tokenizer_sha256": TOKENIZER_SHA256, "anchor_heads_sha256": ANCHOR_SHA256,
        "stimuli_sha256": sha256(stimuli_file), "assay_sha256": sha256(packet / "preflight.json"),
        "protocol_sha256": sha256(protocol), "preflight_sha256": sha256(extractor_preflight),
        "content_forward_per_document": 1, "encoded_documents": len(stimuli),
        "certified_stops": len(stimuli) * ORDINALS, "projection_width": PROJECTION_WIDTH,
        "vectors_shape": [len(stimuli) * ORDINALS, PROJECTION_WIDTH], "compute_dtype": "float32",
        "storage_dtype": "float16", "gpu": "NVIDIA L4", "independent_causal_checks": 12,
    }
    if any(seal.get(key) != value for key, value in expected.items()):
        raise ValueError("Extraction seal does not bind the registered assay, model and protocol")
    for name in ("coordinates.jsonl", "vectors.npy", "causality.json"):
        if sha256(run / name) != seal["file_sha256"][name]:
            raise ValueError(f"Extraction file SHA-256 mismatch for {name}")
    coordinates = read_jsonl(run / "coordinates.jsonl")
    for row, stimulus in zip(coordinates, stimuli, strict=True):
        if row != {"id": stimulus["id"], "stops": stimulus["stops"]}:
            raise ValueError("Extraction axes do not match every frozen stimulus and stop")
    checks = json.loads((run / "causality.json").read_text(encoding="utf-8"))["checks"]
    if len(checks) != 12 or any(row["passed"] is not True or row["projected_tokens"]["passed"] is not True for row in checks):
        raise ValueError("The extraction has incomplete causal certification")
    stored = np.load(run / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if list(stored.shape) != expected["vectors_shape"] or stored.dtype != np.float16:
        raise ValueError("Unexpected stored causal projection axes or storage type")
    return stimuli, stored.reshape(len(stimuli), ORDINALS, PROJECTION_WIDTH)


def assay_bindings(packet: Path, split: Path, extraction: Path, protocol: Path, extractor_preflight: Path) -> dict:
    anchors = {
        "packet_preflight_sha256": sha256(packet / "preflight.json"),
        "split_sha256": sha256(split), "protocol_sha256": sha256(protocol),
        "extractor_preflight_sha256": sha256(extractor_preflight),
    }
    if anchors != {
        "packet_preflight_sha256": ASSAY_SHA256, "split_sha256": SPLIT_SHA256,
        "protocol_sha256": PROTOCOL_SHA256, "extractor_preflight_sha256": EXTRACTOR_PREFLIGHT_SHA256,
    }:
        raise ValueError("Expected the frozen material sample, split, protocol and tokenizer preflight")
    return {
        **anchors,
        "extraction_seal_sha256": sha256(extraction / "seal.json"),
        "trial_source_sha256": TRIAL_TABLE_SHA256, "scorer_sha256": sha256(Path(__file__)),
    }


def stage_fit(packet, split_file, trials, extraction, protocol, extractor_preflight, output):
    if output.exists():
        raise FileExistsError(output)
    seal = assay_bindings(packet, split_file, extraction, protocol, extractor_preflight)
    families = read_packet(packet)
    split = read_split(split_file, packet, families)
    stimuli, stored = extraction_inputs(packet, extraction, protocol, extractor_preflight)
    row_by_id = {row["id"]: index for index, row in enumerate(stimuli)}
    order = np.array([[row_by_id[identifier] for identifier in identifiers] for identifiers in families.values()])
    surfaces = np.stack([surface_features(row["text"], row["stops"]) for row in stimuli])[order]
    states = stored[order].astype(np.float64)
    is_train = np.array([split[family]["split"] == "train" for family in families])

    if sha256(trials) != TRIAL_TABLE_SHA256:
        raise ValueError("Expected the pinned Experiment 1 trial release")
    training, readers = aggregate_trials(trials, split, "train")
    counts = np.array([training[family] for family in families if family in training])
    target_prob = counts / counts.sum(axis=1, keepdims=True)
    position = position_log_masses(training)
    initial_stage = np.array([position[i] - logsumexp(position[i + 1:]) for i in range(ORDINALS)])
    controls, combined, normalization = standardize(surfaces, states, is_train)
    designs = {"surface": controls, "qwen": combined}
    logits = {}
    fit = {
        "normalization": normalization, "ridge": RIDGE,
        "train": inventory(training, readers), "position_log_masses": position, "models": {},
    }
    for name, data in designs.items():
        weights, fit["models"][name] = fit_readout(data[is_train], target_prob, initial_stage)
        logits[name] = data @ weights[ORDINALS:] + weights[:ORDINALS]
        if not np.isfinite(logits[name]).all():
            raise ValueError("Readout produced nonfinite path logits")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".core-vocabulary-fit-") as scratch:
        stage = Path(scratch) / "sealed"
        stage.mkdir()
        write_json(stage / "fit.json", fit)
        with (stage / "forecasts.jsonl").open("x", encoding="utf-8") as f:
            for family_index, identifiers in enumerate(families.values()):
                for draw, identifier in enumerate(identifiers):
                    methods = {name: values[family_index, draw].tolist() for name, values in logits.items()}
                    f.write(json.dumps({"id": identifier, "methods": methods}, allow_nan=False) + "\n")
        seal["file_sha256"] = {name: sha256(stage / name) for name in ("fit.json", "forecasts.jsonl")}
        write_json(stage / "forecast-seal.json", seal)
        stage.rename(output)


def holdout_reader_counts(trials: Path, split: dict, families: list[str]) -> tuple[np.ndarray, int]:
    index_by_family = {family: i for i, family in enumerate(families)}
    outcomes_by_reader = {}
    with trials.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row["exp"], row["stim_set"]) != ("1", "Original"):
                continue
            family = target_family(row["target"])
            if family not in index_by_family:
                continue
            if row["targetWordCondition"] != split[family]["condition"]:
                raise ValueError("Unexpected historical target condition")
            reader = outcomes_by_reader.setdefault(row["subject"], {})
            index = index_by_family[family]
            ordinal = int(row["guesses"])
            correct = row["correct"]
            if (index in reader or not 1 <= ordinal <= ORDINALS or correct not in {"0", "1"}
                    or (correct == "0" and ordinal != ORDINALS)):
                raise ValueError("Invalid held-out historical observation")
            reader[index] = ordinal - 1 if correct == "1" else ORDINALS
    result = np.zeros((len(outcomes_by_reader), len(families), ORDINALS + 1), dtype=np.float64)
    for r, reader in enumerate(sorted(outcomes_by_reader)):
        for family, outcome in outcomes_by_reader[reader].items():
            result[r, family, outcome] = 1
    return result, len(outcomes_by_reader)


def interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def evaluation(masses: np.ndarray, observations: np.ndarray, group_by_family: list[str], draws: int = BOOTSTRAPS) -> dict:
    """Pair methods within fixed reader observations and condition-stratified target draws."""
    totals = observations.sum(axis=0)
    counts = totals.sum(axis=-1, keepdims=True)
    if (counts == 0).any():
        raise ValueError("Every held-out target must have outcomes")
    human_prob = totals / counts
    nll_by_family = -np.einsum("fk,mfk->mf", human_prob, masses)
    model_cdf = np.exp(masses).cumsum(axis=-1)[..., np.asarray(CDF_STEPS) - 1]
    microsteps = np.asarray(CDF_STEPS) - 1
    human_cdf = human_prob.cumsum(axis=-1)[:, microsteps]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = [np.flatnonzero(np.asarray(group_by_family) == condition) for condition in CONDITIONS]
    if any(not len(group) for group in indices):
        raise ValueError("Missing held-out target condition")
    sampled = np.concatenate([rng.choice(group, size=(draws, len(group))) for group in indices], axis=1)
    boot_losses = nll_by_family[:, sampled].mean(axis=-1)
    bootstrap_target_cdf = human_cdf[sampled].mean(axis=1)
    bootstrap_crossed_cdf = np.empty_like(bootstrap_target_cdf)
    bootstrap_model_cdf = model_cdf[:, sampled].mean(axis=-2)
    crossed_losses = np.empty_like(boot_losses)
    n_readers, n_targets, n_outcomes = observations.shape
    uniform_readers = np.full(n_readers, 1 / n_readers)
    flat = observations.reshape(n_readers, -1)
    for start in range(0, draws, 128):
        stop = min(start + 128, draws)
        weights = rng.multinomial(n_readers, uniform_readers, size=stop - start)
        weighted = (weights @ flat).reshape(stop - start, n_targets, n_outcomes)
        at_targets = weighted[np.arange(stop - start)[:, None], sampled[start:stop]]
        total_weights = at_targets.sum(axis=-1, keepdims=True)
        if (total_weights == 0).any():
            raise ValueError("Crossed bootstrap lost all observations for a sampled target")
        probabilities = at_targets / total_weights
        bootstrap_crossed_cdf[start:stop] = probabilities.cumsum(axis=-1)[..., microsteps].mean(axis=1)
        crossed_losses[:, start:stop] = (
            -np.einsum("dfk,mdfk->md", probabilities, masses[:, sampled[start:stop]]) / n_targets
        )

    comparisons = {}
    for i, control in enumerate(METHODS[:-1]):
        difference = interval(boot_losses[i] - boot_losses[-1])
        comparisons[control] = {
            "control_minus_qwen_nll_nats": float((nll_by_family[i] - nll_by_family[-1]).mean()),
            "paired_target_95_ci": difference, "passes": difference[0] > 0,
            "paired_crossed_95_ci": interval(crossed_losses[i] - crossed_losses[-1]),
        }
    cdf_by_method = {}
    for i, method in enumerate(METHODS):
        entries = []
        for j, step in enumerate(CDF_STEPS):
            difference = model_cdf[i, :, j] - human_cdf[:, j]
            point = float(difference.mean())
            confidence = interval(bootstrap_model_cdf[i, :, j] - bootstrap_crossed_cdf[:, j])
            entries.append({
                "exposure": step, "human_cdf": float(human_cdf[:, j].mean()),
                "human_crossed_95_ci": interval(bootstrap_crossed_cdf[:, j]),
                "model_cdf": float(model_cdf[i, :, j].mean()), "signed_model_minus_human": point,
                "paired_target_95_ci": interval(bootstrap_model_cdf[i, :, j] - bootstrap_target_cdf[:, j]),
                "paired_crossed_95_ci": confidence,
                "within_margin": abs(point) < CDF_MARGIN and confidence[0] > -CDF_MARGIN and confidence[1] < CDF_MARGIN,
            })
        cdf_by_method[method] = entries
    return {
        "target_macro_nll_nats": {name: float(nll_by_family[i].mean()) for i, name in enumerate(METHODS)},
        "paired_nll_superiority": comparisons, "passes_nll": all(row["passes"] for row in comparisons.values()),
        "cdf_margin": CDF_MARGIN, "cdf_calibration": cdf_by_method,
        "passes_qwen_cdf": all(row["within_margin"] for row in cdf_by_method["qwen"]),
        "bootstrap": {
            "draws": draws, "seed": BOOTSTRAP_SEED, "interval": 0.95,
            "target": "paired target resampling within condition, conditional on observed readers",
            "crossed": "shared readers crossed with paired target resampling within condition",
            "nll_gate_interval": "target", "cdf_gate_interval": "crossed",
        },
    }


def stage_score(packet, split_file, trials, extraction, protocol, extractor_preflight, fit_dir, output):
    if output.exists():
        raise FileExistsError(output)
    binding = assay_bindings(packet, split_file, extraction, protocol, extractor_preflight)
    seal_file = fit_dir / "forecast-seal.json"
    seal = json.loads(seal_file.read_text(encoding="utf-8"))
    if seal.keys() != binding.keys() | {"file_sha256"} or any(seal[key] != value for key, value in binding.items()):
        raise ValueError("Forecast seal binding changed")
    if seal["file_sha256"] != {name: sha256(fit_dir / name) for name in ("fit.json", "forecasts.jsonl")}:
        raise ValueError("Forecast or fitted parameter bytes changed after sealing")
    families = read_packet(packet)
    split = read_split(split_file, packet, families)
    predictors = forecast_mixtures(fit_dir / "forecasts.jsonl", families)
    if set(predictors) != set(METHODS[1:]):
        raise ValueError("Unexpected ordered comparisons")
    position = json.loads((fit_dir / "fit.json").read_text(encoding="utf-8"))["position_log_masses"]
    holdout = [family for family in families if split[family]["split"] == "holdout"]
    predictors["position"] = dict.fromkeys(holdout, position)
    masses = np.asarray([[predictors[method][family] for family in holdout] for method in METHODS])
    if not np.isfinite(masses).all() or not np.allclose(logsumexp(masses, axis=-1), 0, rtol=0, atol=1e-12):
        raise ValueError("A sealed forecast has invalid outcome probabilities")

    if sha256(trials) != TRIAL_TABLE_SHA256:
        raise ValueError("Expected the pinned Experiment 1 trial release")
    observed, readers = holdout_reader_counts(trials, split, holdout)
    result = evaluation(masses, observed, [split[family]["condition"] for family in holdout])
    counts = {family: row.astype(int).tolist() for family, row in zip(holdout, observed.sum(axis=0), strict=True)}
    result["holdout"] = inventory(counts, readers)
    result["forecast_seal_sha256"] = sha256(seal_file)
    write_json(output, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "score"):
        arguments = sub.add_parser(command)
        for option in ("packet", "split", "trials", "extraction", "protocol", "extractor-preflight", "output"):
            arguments.add_argument(f"--{option}", type=Path, required=True)
        if command == "score":
            arguments.add_argument("--fit", type=Path, required=True)
    args = parser.parse_args()
    common = (args.packet, args.split, args.trials, args.extraction, args.protocol, args.extractor_preflight)
    if args.command == "fit":
        stage_fit(*common, args.output)
    else:
        stage_score(*common, args.fit, args.output)


if __name__ == "__main__":
    main()
