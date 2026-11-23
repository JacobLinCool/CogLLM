"""Replay the frozen Core Vocabulary readouts on disjoint grade-free cue paths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import logsumexp

from research.semantic_tracking.beukers_extract import FP16_TOLERANCE, read_stimuli
from research.semantic_tracking.core_vocabulary_disjoint_panel import (
    CAUSAL_PREFLIGHT_FILE, MIXTURE_FILE, ORIGINAL_DRAWS, PANEL_DRAWS, SPLIT_SHA256,
    ORIGINAL_PREFLIGHT_SHA256, STIMULI_FILE,
)
from research.semantic_tracking.core_vocabulary_marginal_fit import (
    BOOTSTRAPS, BOOTSTRAP_SEED, CDF_STEPS, METHODS, PROJECTION_WIDTH, SURFACE,
    evaluation, extraction_inputs, holdout_reader_counts, interval, surface_features,
)
from research.semantic_tracking.core_vocabulary_marginal_prepare import ORDINALS
from research.semantic_tracking.core_vocabulary_marginal_score import (
    CONDITIONS, TRIAL_TABLE_SHA256, forecast_mixtures, inventory, read_jsonl, read_packet,
    read_split, sha256, write_json,
)


PROTOCOL_SHA256 = "8f7f4ef379f89ac79ac7a343ee4dc11b52215b9831a87f4c19d819a8cbb3837d"
PANEL_SHA256 = "20f5e6dfbc9e5936d5ba6a5701e14a9810200d356314d94145f04030d02b6edf"
CAUSAL_PREFLIGHT_SHA256 = "69dafdd439abf372455b7a29e6159769660bfbbb3155ce3f1b3cff4be025e8f1"
ARCHIVED_SHA256 = {
    "fit.json": "b18f5d5beda601eb725621d7299331678de33dbeae68b18b9c9db01bcbd5cd91",
    "forecasts.jsonl": "3a5269b946fa8aed03b0359c09684f2bbabe05c790af4d50f426c01045551520",
    "forecast-seal.json": "56466f6d2a282184205142ac30df71482248268f5b3ff08ab81575f157dac06c",
    "extraction-seal.json": "d3ff741efc862d3fd1bdcd8cf769268ff9a5469b782fea60b74bcbde923e986c",
}
PANELS = ("original8", "disjoint32", "pooled40")
FORECAST_FILE = "forecasts.jsonl"
SEAL_FILE = "seal.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bind_inputs(panel: Path, original_packet: Path, split: Path, archived: Path, protocol: Path,
                extraction: Path) -> dict:
    frozen = {
        panel / "preflight.json": PANEL_SHA256,
        panel / CAUSAL_PREFLIGHT_FILE: CAUSAL_PREFLIGHT_SHA256,
        original_packet / "preflight.json": ORIGINAL_PREFLIGHT_SHA256,
        split: SPLIT_SHA256,
        protocol: PROTOCOL_SHA256,
        **{archived / filename: digest for filename, digest in ARCHIVED_SHA256.items()},
    }
    for path, expected in frozen.items():
        if sha256(path) != expected:
            raise ValueError(f"Unexpected frozen input SHA-256: {path.name}")

    panel_manifest = read_json(panel / "preflight.json")
    if any(sha256(panel / name) != digest for name, digest in panel_manifest["file_sha256"].items()):
        raise ValueError("A grade-free disjoint panel artifact changed")

    original_seal = read_json(archived / "extraction-seal.json")
    new_seal = read_json(extraction / SEAL_FILE)
    frozen_features = ("candidate_id", "model", "revision", "tokenizer_sha256", "transformers",
                       "anchor_heads_sha256", "projection_sha256", "projection_width", "source_code_sha256",
                       "gpu", "compute_dtype", "storage_dtype")
    if any(new_seal.get(key) != original_seal[key] for key in frozen_features):
        raise ValueError("Extraction uses a foreign candidate, feature projection or executable")
    causal = read_json(panel / CAUSAL_PREFLIGHT_FILE)
    if (any(new_seal.get(key) != value for key, value in causal.items())
            or new_seal.get("assay_sha256") != PANEL_SHA256
            or new_seal.get("preflight_sha256") != CAUSAL_PREFLIGHT_SHA256
            or new_seal.get("protocol_sha256") != PROTOCOL_SHA256
            or new_seal.get("content_batches") != 32
            or not 0 <= new_seal.get("max_storage_relative_l2", math.inf) <= FP16_TOLERANCE
            or {"raw_stops", "states_shape", "terminal_states_shape"}.intersection(new_seal)):
        raise ValueError("Extraction differs from the frozen projection-only six-stop assay")
    expected = {"coordinates.jsonl", "vectors.npy", "causality.json", "telemetry.jsonl"}
    if (new_seal["file_sha256"].keys() != expected
            or any(sha256(extraction / name) != new_seal["file_sha256"][name] for name in expected)):
        raise ValueError("Extraction artifacts changed or contain unexpected feature types")
    return {
        "schema_version": 1,
        "disjoint_panel_sha256": PANEL_SHA256,
        "disjoint_causal_preflight_sha256": CAUSAL_PREFLIGHT_SHA256,
        "extraction_seal_sha256": sha256(extraction / SEAL_FILE),
        "protocol_sha256": PROTOCOL_SHA256,
        "split_sha256": SPLIT_SHA256,
        "original_packet_sha256": ORIGINAL_PREFLIGHT_SHA256,
        "original_artifact_sha256": ARCHIVED_SHA256,
        "trial_source_sha256": TRIAL_TABLE_SHA256,
        "scorer_sha256": sha256(Path(__file__)),
    }


def panel_material(panel: Path, original_packet: Path, split_file: Path) -> tuple:
    original_families = read_packet(original_packet)
    split = read_split(split_file, original_packet, original_families)
    heldout = {key: ids for key, ids in original_families.items() if split[key]["split"] == "holdout"}
    stimuli = read_stimuli(panel / STIMULI_FILE)
    rows = read_jsonl(panel / MIXTURE_FILE)
    frozen_order = [key for key, row in split.items() if row["split"] == "holdout"]
    expected = [(family, draw) for family in frozen_order for draw in range(1, PANEL_DRAWS + 1)]
    families = {key: [] for key in frozen_order}
    original_texts = read_stimuli(original_packet / STIMULI_FILE)
    original_ids = {row["id"] for row in original_texts}
    original_contents = {row["text"] for row in original_texts}
    for stimulus, row, (family, draw) in zip(stimuli, rows, expected, strict=True):
        if (tuple(stimulus) != ("id", "text", "stops") or tuple(row) != ("id", "family", "draw", "weight")
                or row != {"id": stimulus["id"], "family": family, "draw": draw, "weight": 1 / PANEL_DRAWS}
                or stimulus["id"] in original_ids or stimulus["text"] in original_contents
                or len(stimulus["stops"]) != ORDINALS or stimulus["stops"][-1] != len(stimulus["text"])):
            raise ValueError("Unexpected, reordered or leaking disjoint stimulus or family key")
        families[family].append(stimulus["id"])
    return stimuli, families, original_texts, original_families, heldout, split


def frozen_feature_blocks(fit: dict, surface: np.ndarray, states: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the archived training transforms to any aligned surface and projection axes."""
    if (surface.ndim < 2 or states.ndim != surface.ndim or states.shape[:-1] != surface.shape[:-1]
            or surface.shape[-1] != len(SURFACE) or states.shape[-1] != PROJECTION_WIDTH):
        raise ValueError("Surface and projection arrays must share their leading axes")
    norms = np.linalg.norm(states.astype(np.float64), axis=-1, keepdims=True)
    if not np.allclose(norms, 1, rtol=0, atol=0.002):
        raise ValueError("Projected states must be finite and approximately unit norm")
    norm = fit["normalization"]
    if norm["surface_features"] != list(SURFACE):
        raise ValueError("Frozen surface feature order differs")
    surface = (surface - norm["surface_means"]) / norm["surface_scales"]
    qwen = (states / norms - norm["qwen_unit_vector_mean"]) / norm["qwen_centered_rms_l2"]
    return {"surface": surface, "qwen": np.concatenate((surface, qwen), axis=-1)}


def apply_frozen_fit(fit: dict, stimuli: list[dict], states: np.ndarray) -> dict[str, np.ndarray]:
    expected = (len(stimuli), ORDINALS, PROJECTION_WIDTH)
    if states.shape != expected or states.dtype != np.float16:
        raise ValueError("Expected one FP16 projection per exact stimulus stop")
    surface = np.stack([surface_features(row["text"], row["stops"]) for row in stimuli])
    features = frozen_feature_blocks(fit, surface, states)
    logits = {name: values @ fit["models"][name]["slopes"] + fit["models"][name]["intercepts"]
              for name, values in features.items()}
    if any(not np.isfinite(values).all() for values in logits.values()):
        raise ValueError("Frozen fit produced nonfinite path logits")
    return logits


def read_ordered_logits(path: Path, stimuli: list[dict]) -> dict[str, np.ndarray]:
    logits = {method: [] for method in METHODS[1:]}
    for row, stimulus in zip(read_jsonl(path), stimuli, strict=True):
        if tuple(row) != ("id", "methods") or row["id"] != stimulus["id"] or tuple(row["methods"]) != METHODS[1:]:
            raise ValueError("Forecast axes must exactly match the grade-free input order and frozen methods")
        for method, values in row["methods"].items():
            if len(values) != ORDINALS or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
                raise ValueError("Each method needs six finite hazard logits")
            logits[method].append(values)
    return {method: np.asarray(values, dtype=np.float64) for method, values in logits.items()}


def original_replay(archived: Path, fit: dict, stimuli: list[dict]) -> dict[str, float]:
    seal = read_json(archived / "extraction-seal.json")
    for name in ("vectors.npy", "coordinates.jsonl"):
        if sha256(archived / name) != seal["file_sha256"][name]:
            raise ValueError(f"Original extraction artifact changed: {name}")
    if read_jsonl(archived / "coordinates.jsonl") != [{"id": row["id"], "stops": row["stops"]} for row in stimuli]:
        raise ValueError("Original extraction axes disagree with the canonical randomization")
    states = np.load(archived / "vectors.npy", mmap_mode="r", allow_pickle=False)
    calculated = apply_frozen_fit(fit, stimuli, states.reshape(len(stimuli), ORDINALS, PROJECTION_WIDTH))
    expected = read_ordered_logits(archived / FORECAST_FILE, stimuli)
    errors = {method: float(np.max(np.abs(calculated[method] - values))) for method, values in expected.items()}
    if max(errors.values()) > 1e-12:
        raise ValueError("Frozen CPU replay disagrees with the canonical path forecasts")
    return errors


def stage_forecast(panel, original_packet, split, archived, protocol, extraction, output):
    if output.exists():
        raise FileExistsError(output)
    binding = bind_inputs(panel, original_packet, split, archived, protocol, extraction)
    stimuli, _, prior, _, _, _ = panel_material(panel, original_packet, split)
    fit = read_json(archived / "fit.json")
    errors = original_replay(archived, fit, prior)
    certified, states = extraction_inputs(panel, extraction, protocol, panel / CAUSAL_PREFLIGHT_FILE)
    coordinates = read_jsonl(extraction / "coordinates.jsonl")
    if certified != stimuli or any(tuple(row) != ("id", "stops") for row in coordinates):
        raise ValueError("Extracted inputs and the model-ready panel disagree")
    logits = apply_frozen_fit(fit, stimuli, states)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".core-vocabulary-disjoint-") as scratch:
        stage = Path(scratch) / "sealed"
        stage.mkdir()
        with (stage / FORECAST_FILE).open("x", encoding="utf-8", newline="\n") as stream:
            for index, row in enumerate(stimuli):
                methods = {method: logits[method][index].tolist() for method in METHODS[1:]}
                stream.write(json.dumps({"id": row["id"], "methods": methods}, allow_nan=False) + "\n")
        seal = {**binding, "forecast_sha256": sha256(stage / FORECAST_FILE), "original_logit_max_abs_error": errors}
        write_json(stage / SEAL_FILE, seal)
        stage.rename(output)


def panel_masses(fit: dict, original_forecasts: Path, original_families: dict,
                 disjoint_forecasts: Path, disjoint_families: dict, heldout: dict) -> np.ndarray:
    archived = forecast_mixtures(original_forecasts, original_families)
    disjoint = forecast_mixtures(disjoint_forecasts, disjoint_families)
    position = np.broadcast_to(fit["position_log_masses"], (len(heldout), ORDINALS + 1))
    masses = [np.asarray([position, *[[panel[name][key] for key in heldout] for name in METHODS[1:]]])
              for panel in (archived, disjoint)]
    paths = ORIGINAL_DRAWS + PANEL_DRAWS
    pooled = np.logaddexp(masses[0] + math.log(ORIGINAL_DRAWS / paths), masses[1] + math.log(PANEL_DRAWS / paths))
    pooled[0] = position
    result = np.asarray([*masses, pooled])
    if not np.isfinite(result).all() or not np.allclose(logsumexp(result, axis=-1), 0, rtol=0, atol=1e-12):
        raise ValueError("Every panel, target and method must have a finite, normalized seven-outcome distribution")
    return result


def panel_changes(masses: np.ndarray, observations: np.ndarray, conditions: list[str],
                  draws: int = BOOTSTRAPS) -> dict:
    """Compare each new panel with the original using shared family and reader draws."""
    contrast = masses[1:] - masses[0]
    people, families, outcomes = observations.shape
    totals = observations.sum(axis=0)
    human = totals / totals.sum(axis=-1, keepdims=True)
    by_family = -np.einsum("fk,pmfk->pmf", human, contrast)
    point = by_family.mean(axis=-1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = [np.flatnonzero(np.asarray(conditions) == condition) for condition in CONDITIONS]
    sampled = np.concatenate([rng.choice(group, size=(draws, len(group))) for group in groups], axis=1)
    target_draws = by_family[:, :, sampled].mean(axis=-1)
    crossed_draws = np.empty_like(target_draws)
    uniform = np.full(people, 1 / people)
    for start in range(0, draws, 128):
        stop = min(start + 128, draws)
        weights = rng.multinomial(people, uniform, size=stop - start)
        probabilities = (weights @ observations.reshape(people, -1)).reshape(-1, families, outcomes)
        current = sampled[start:stop]
        probabilities = probabilities[np.arange(stop - start)[:, None], current]
        denominators = probabilities.sum(axis=-1, keepdims=True)
        if (denominators == 0).any():
            raise ValueError("Crossed bootstrap lost the observations for a sampled family")
        probabilities = probabilities / denominators
        crossed_draws[:, :, start:stop] = -np.einsum("dfk,pmdfk->pmd", probabilities, contrast[:, :, current]) / families

    def report(value, target, crossed):
        return {"point": float(value), "paired_target_95_ci": interval(target),
                "paired_crossed_95_ci": interval(crossed)}

    cdf = np.exp(masses).cumsum(axis=-1)[..., np.asarray(CDF_STEPS) - 1]
    cdf_change = cdf[1:] - cdf[0]
    cdf_draws = cdf_change[:, :, sampled].mean(axis=-2)
    return {
        name: {
            "nll_nats_new_minus_original": {
                method: report(point[i, m], target_draws[i, m], crossed_draws[i, m])
                for m, method in enumerate(METHODS)
            },
            "gain_over_surface_nll_nats_new_minus_original": report(
                point[i, 1] - point[i, 2], target_draws[i, 1] - target_draws[i, 2],
                crossed_draws[i, 1] - crossed_draws[i, 2],
            ),
            "cdf_new_minus_original": {
                method: [{"exposure": step, "point": float(cdf_change[i, m, :, j].mean()),
                          "paired_target_95_ci": interval(cdf_draws[i, m, :, j])}
                         for j, step in enumerate(CDF_STEPS)]
                for m, method in enumerate(METHODS)
            },
        }
        for i, name in enumerate(PANELS[1:])
    }


def stage_score(panel, original_packet, split, archived, protocol, extraction, output, forecasts, trials):
    if output.exists():
        raise FileExistsError(output)
    binding = bind_inputs(panel, original_packet, split, archived, protocol, extraction)
    seal = read_json(forecasts / SEAL_FILE)
    if (seal.keys() != binding.keys() | {"forecast_sha256", "original_logit_max_abs_error"}
            or any(seal[key] != value for key, value in binding.items())
            or seal["forecast_sha256"] != sha256(forecasts / FORECAST_FILE)
            or set(seal["original_logit_max_abs_error"]) != set(METHODS[1:])
            or any(not 0 <= value <= 1e-12 for value in seal["original_logit_max_abs_error"].values())):
        raise ValueError("Disjoint forecasts must be sealed against the exact sources before joining outcomes")
    stimuli, new, prior, original_families, heldout, frozen_split = panel_material(panel, original_packet, split)
    read_ordered_logits(forecasts / FORECAST_FILE, stimuli)
    read_ordered_logits(archived / FORECAST_FILE, prior)
    masses = panel_masses(read_json(archived / "fit.json"), archived / FORECAST_FILE, original_families,
                         forecasts / FORECAST_FILE, new, heldout)
    if sha256(trials) != TRIAL_TABLE_SHA256:
        raise ValueError("Expected the pinned historical trial table")
    observations, readers = holdout_reader_counts(trials, frozen_split, list(heldout))
    counts = {family: row.astype(int).tolist() for family, row in zip(heldout, observations.sum(axis=0), strict=True)}
    summary = inventory(counts, readers)
    if (summary["targets"], summary["trials"]) != (32, 1603):
        raise ValueError("Outcome inventory differs from the canonical holdout")
    groups = [frozen_split[family]["condition"] for family in heldout]
    result = {
        "holdout": summary,
        "forecast_seal_sha256": sha256(forecasts / SEAL_FILE),
        "primary_panel": PANELS[1],
        "panels": {name: evaluation(values, observations, groups) for name, values in zip(PANELS, masses, strict=True)},
        "paired_changes_from_original8": panel_changes(masses, observations, groups),
    }
    write_json(output, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("forecast", "score"):
        stage = commands.add_parser(command)
        for option in ("panel", "original-packet", "split", "archived", "protocol", "extraction", "output"):
            stage.add_argument(f"--{option}", type=Path, required=True)
        if command == "score":
            stage.add_argument("--forecasts", type=Path, required=True)
            stage.add_argument("--trials", type=Path, required=True)
    args = vars(parser.parse_args())
    (stage_forecast if args.pop("command") == "forecast" else stage_score)(**args)


if __name__ == "__main__":
    main()
