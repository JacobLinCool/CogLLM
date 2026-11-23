"""Run the registered ACT and Core Vocabulary joint hazard comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import logsumexp

from research.semantic_tracking import act_transfer_score as act
from research.semantic_tracking import core_vocabulary_disjoint_score as disjoint
from research.semantic_tracking.core_vocabulary_marginal_fit import (
    holdout_reader_counts, surface_features,
)
from research.semantic_tracking.core_vocabulary_marginal_score import (
    CONDITIONS, TRIAL_TABLE_SHA256, aggregate_trials, read_jsonl, sha256, write_json,
)
from research.semantic_tracking.joint_cue_hazard import METHODS, HazardTraining, fit_hazards, forecast_log_masses


PROTOCOL = Path(__file__).with_name("joint-cue-hazard-protocol.md")
PROTOCOL_SHA256 = "25e169b7a98d09942fdf4a802aade0d37a28030508894e2268957bee4d85ec6a"
ACT_ARCHIVE = "act-transfer-l4"
CORE_ARCHIVE = "core-vocabulary-qwen-l4"
DISJOINT_ARCHIVE = "core-vocabulary-disjoint-l4"
ARCHIVE_PINS = {
    ACT_ARCHIVE: "dca1735627761f7da38078a254c4d6259984554a1f387e580a51012f59faf6d7",
    CORE_ARCHIVE: disjoint.ARCHIVED_SHA256["extraction-seal.json"],
    DISJOINT_ARCHIVE: "e81aaa01780fc5fc68373563c7bbbc75a94bb8f8cf431d39753e3b3537c26e21",
}
ACT = "act_german"
CORE = "core_vocabulary"
HORIZON = act.PRIMARY_HORIZON


@dataclass(frozen=True)
class Cues:
    families: tuple[str, ...]
    surface: np.ndarray
    representation: np.ndarray

    def select(self, mask: np.ndarray) -> Cues:
        names = tuple(key for key, keep in zip(self.families, mask, strict=True) if keep)
        return Cues(names, self.surface[mask], self.representation[mask])


@dataclass(frozen=True)
class Materials:
    core_train: Cues
    core_holdout: Cues
    core_split: dict
    act_cues: Cues
    act_rows: list[dict]
    binding: dict


def read_projection(directory: Path, stimuli: list[dict], identity: dict) -> np.ndarray:
    seal_file = directory / "extraction-seal.json"
    act.checked(seal_file, ARCHIVE_PINS[directory.name])
    seal = disjoint.read_json(seal_file)
    if any(seal[name] != value for name, value in identity.items()):
        raise ValueError("All cue features must come from the same candidate and projection")
    for name in ("coordinates.jsonl", "vectors.npy", "causality.json", "telemetry.jsonl"):
        act.checked(directory / name, seal["file_sha256"][name])
    coordinates = read_jsonl(directory / "coordinates.jsonl")
    if coordinates != [{"id": row["id"], "stops": row["stops"]} for row in stimuli]:
        raise ValueError("Cue feature order or exact stop changed")
    sizes = {len(row["stops"]) for row in stimuli}
    vectors = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if (len(sizes) != 1 or list(vectors.shape) != seal["vectors_shape"] or vectors.dtype != np.float16
            or vectors.shape != (sum(len(row["stops"]) for row in stimuli), identity["projection_width"])):
        raise ValueError("Malformed projected cue coordinates")
    return vectors.reshape(len(stimuli), sizes.pop(), identity["projection_width"])[:, :HORIZON]


def surface_rows(row: dict) -> np.ndarray:
    stops = row["stops"][:HORIZON]
    text = row["text"][:stops[-1]]
    lengths = np.asarray([len(line) for line in text.split("\n")])
    return np.column_stack((surface_features(text, stops), np.log1p(lengths), np.log1p(lengths.cumsum())))


def group_cues(stimuli: list[dict], states: np.ndarray, families: dict) -> Cues:
    by_id = {row["id"]: i for i, row in enumerate(stimuli)}
    order = np.asarray([[by_id[key] for key in paths] for paths in families.values()])
    surface = np.stack([surface_rows(row) for row in stimuli])[order]
    vectors = states[order].astype(np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.allclose(norms, 1.0, rtol=0, atol=0.002):
        raise ValueError("The frozen projection is not unit-normalized within its storage tolerance")
    return Cues(tuple(families), surface, vectors / norms)


def read_materials(source: Path, archives: Path) -> Materials:
    act.checked(PROTOCOL, PROTOCOL_SHA256)
    core_source = source / "core-vocabulary"
    pilot = core_source / "marginal-pilot-v1"
    panel = core_source / "marginal-disjoint-panel-v1"
    split_file = core_source / "marginal-score-v1/split.json"
    for path, digest in ((pilot / "preflight.json", disjoint.ORIGINAL_PREFLIGHT_SHA256),
                         (panel / "preflight.json", disjoint.PANEL_SHA256),
                         (split_file, disjoint.SPLIT_SHA256)):
        act.checked(path, digest)
    for name, digest in disjoint.read_json(panel / "preflight.json")["file_sha256"].items():
        act.checked(panel / name, digest)
    new_text, new_families, old_text, old_families, holdout, core_split = disjoint.panel_material(panel, pilot, split_file)
    if set(new_families) != set(holdout):
        raise ValueError("Disjoint paths must cover exactly the original holdout families")
    training = {f: paths for f, paths in old_families.items() if core_split[f]["split"] == "train"}

    act_material = source / "act-german/material-v1"
    act.checked(act_material / "preflight.json", act.MATERIAL_SHA256)
    act.checked(act_material / "split.json", act.SPLIT_SHA256)
    for name, digest in disjoint.read_json(act_material / "preflight.json")["file_sha256"].items():
        act.checked(act_material / name, digest)
    act_text = read_jsonl(act_material / act.STIMULI_FILE)
    act_rows = disjoint.read_json(act_material / "split.json")["material"]
    if ([x["id"] for x in act_text] != [x["id"] for x in act_rows]
            or np.bincount([x["fold"] for x in act_rows]).tolist() != [4] * act.FOLDS):
        raise ValueError("ACT materials and frozen folds are not aligned")
    act_families = {row["family"]: (row["id"],) for row in act_rows}

    original = disjoint.read_json(archives / CORE_ARCHIVE / "extraction-seal.json")
    identity = {name: original[name] for name in (
        "candidate_id", "model", "revision", "projection_sha256", "tokenizer_sha256", "projection_width",
    )}
    old_states = read_projection(archives / CORE_ARCHIVE, old_text, identity)
    new_states = read_projection(archives / DISJOINT_ARCHIVE, new_text, identity)
    act_states = read_projection(archives / ACT_ARCHIVE, act_text, identity)
    binding = {
        "protocol_sha256": PROTOCOL_SHA256, "analysis_sha256": sha256(Path(__file__)),
        "solver_sha256": sha256(Path(__file__).with_name("joint_cue_hazard.py")),
        "extractors_sha256": ARCHIVE_PINS, "identity": identity,
        "core_material_sha256": disjoint.ORIGINAL_PREFLIGHT_SHA256, "core_split_sha256": disjoint.SPLIT_SHA256,
        "core_grade_sha256": TRIAL_TABLE_SHA256, "new_core_paths_sha256": disjoint.PANEL_SHA256,
        "act_material_sha256": act.MATERIAL_SHA256, "act_split_sha256": act.SPLIT_SHA256,
        "act_grade_sha256": act.OUTCOMES_SHA256,
    }
    return Materials(group_cues(old_text, old_states, training), group_cues(new_text, new_states, new_families),
                     core_split, group_cues(act_text, act_states, act_families), act_rows, binding)


def normalizer(cues: Cues) -> dict:
    surface_center = cues.surface.mean(axis=(0, 1))
    constant = np.ptp(cues.surface, axis=(0, 1)) == 0
    surface_center = np.where(constant, cues.surface[0, 0], surface_center)
    cue_scale = np.sqrt(np.mean((cues.surface - surface_center) ** 2, axis=(0, 1, 2)))
    representation_center = cues.representation.mean(axis=(0, 1))
    residual = cues.representation - representation_center
    representation_scale = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=-1))))
    if not np.isfinite(representation_scale) or representation_scale <= 0:
        raise ValueError("Training needs within-exposure variation in the frozen representation")
    return {"surface_center": surface_center.tolist(), "surface_scale": np.where(cue_scale == 0, 1, cue_scale).tolist(),
            "representation_center": representation_center.tolist(), "representation_scale": representation_scale}


def normalize(cues: Cues, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    surface = (cues.surface - stats["surface_center"]) / stats["surface_scale"]
    representation = (cues.representation - stats["representation_center"]) / stats["representation_scale"]
    return surface, representation


def fit_fold(core: HazardTraining, act_cues: Cues, counts: np.ndarray, train: np.ndarray) -> tuple[dict, dict, dict]:
    cues = act_cues.select(train)
    stats = normalizer(cues)
    training = {CORE: core, ACT: HazardTraining(*normalize(cues, stats), counts[train])}
    fits, diagnostics = {}, {}
    for method in METHODS:
        fits[method], diagnostics[method] = fit_hazards(training, method)
    return fits, stats, diagnostics


def cue_predictions(models: dict, study: str, features: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return np.stack([forecast_log_masses(models[method][study], *features) for method in METHODS])


def stage_forecast(source: Path, archives: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    data = read_materials(source, archives)
    trials = source / "core-vocabulary/raw/data_trials.csv"
    act.checked(trials, TRIAL_TABLE_SHA256)
    core_counts, readers = aggregate_trials(trials, data.core_split, "train")
    core_stats = normalizer(data.core_train)
    core = HazardTraining(*normalize(data.core_train, core_stats), np.array([core_counts[k] for k in data.core_train.families]))
    if readers != 195 or int(core.counts.sum()) != 3077:
        raise ValueError("English training inventory changed")
    core_features = normalize(data.core_holdout, core_stats)
    outcomes, _ = act.observations(source / "act-german/assessment-v1", data.act_rows)
    act_counts = outcomes.sum(axis=0)
    folds = np.array([row["fold"] for row in data.act_rows])
    core_predictions = []
    act_predictions = np.full((len(METHODS), len(data.act_rows), HORIZON + 1), np.nan)
    models = {"core_normalization": core_stats, "folds": []}
    for fold in range(act.FOLDS):
        training = folds != fold
        fit, stats, diagnostics = fit_fold(core, data.act_cues, act_counts, training)
        heldout = normalize(data.act_cues.select(~training), stats)
        act_predictions[:, ~training] = cue_predictions(fit, ACT, heldout)
        core_predictions.append(cue_predictions(fit, CORE, core_features))
        models["folds"].append({"fold": fold, "act_normalization": stats, "readouts": fit, "diagnostics": diagnostics})
    core_ensemble = logsumexp(np.stack(core_predictions), axis=0) - np.log(act.FOLDS)
    predictions = {ACT: (data.act_cues.families, act_predictions), CORE: (data.core_holdout.families, core_ensemble)}
    for _, masses in predictions.values():
        if not np.isfinite(masses).all() or not np.allclose(logsumexp(masses, axis=-1), 0, rtol=0, atol=1e-12):
            raise ValueError("A cross-fitted cue distribution is invalid")
    output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(prefix=".joint-cue-hazard-", dir=output.parent) as temporary:
        sealed = Path(temporary) / "sealed"
        sealed.mkdir()
        write_json(sealed / "models.json", models)
        with (sealed / "forecasts.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for study, (families, masses) in predictions.items():
                for i, family in enumerate(families):
                    stream.write(json.dumps({"study": study, "family": family,
                                             "log_masses": {m: masses[j, i].tolist() for j, m in enumerate(METHODS)}},
                                            allow_nan=False) + "\n")
        write_json(sealed / "seal.json", {**data.binding, "file_sha256": {
            name: sha256(sealed / name) for name in ("models.json", "forecasts.jsonl")
        }})
        sealed.rename(output)


def evaluate(masses: np.ndarray, outcomes: np.ndarray, strata: list[np.ndarray]) -> dict:
    n_readers, n_items, _ = outcomes.shape
    totals = outcomes.sum(axis=0)
    probs = totals / totals.sum(axis=-1, keepdims=True)
    loss = -np.einsum("ik,mik->mi", probs, masses)
    steps = np.asarray(act.PRIMARY_STEPS) - 1
    truth_cdf = probs.cumsum(axis=-1)[:, steps]
    model_cdf = np.exp(masses[-1]).cumsum(axis=-1)[:, steps]

    rng = np.random.default_rng(act.BOOTSTRAP_SEED)
    draws = act.BOOTSTRAPS
    sampled = np.concatenate([rng.choice(group, size=(draws, len(group))) for group in strata], axis=1)
    loss_by_item = loss[:, sampled].mean(axis=-1)
    loss_crossed = np.empty_like(loss_by_item)
    sampled_model_cdf = model_cdf[sampled].mean(axis=-2)
    sampled_truth_cdf = truth_cdf[sampled].mean(axis=-2)
    crossed_truth_cdf = np.empty_like(sampled_truth_cdf)
    readers = np.full(n_readers, 1 / n_readers)
    flat = outcomes.reshape(n_readers, -1)
    for start in range(0, draws, 128):
        stop = min(start + 128, draws)
        reader_weights = rng.multinomial(n_readers, readers, size=stop - start)
        reader_counts = (reader_weights @ flat).reshape(-1, *probs.shape)
        selected = sampled[start:stop]
        counts = reader_counts[np.arange(stop - start)[:, None], selected]
        n = counts.sum(axis=-1, keepdims=True)
        if (n == 0).any():
            raise ValueError("Crossed resampling lost all observations for one material")
        resampled = counts / n
        loss_crossed[:, start:stop] = -np.einsum("dik,mdik->md", resampled, masses[:, selected]) / n_items
        crossed_truth_cdf[start:stop] = resampled.cumsum(axis=-1)[..., steps].mean(axis=-2)

    comparisons = {}
    for i, name in enumerate(METHODS[:-1]):
        interval = act.interval(loss_by_item[i] - loss_by_item[-1])
        comparisons[name] = {"control_minus_joint_nll_nats": float((loss[i] - loss[-1]).mean()),
                             "paired_material_95_ci": interval,
                             "paired_crossed_95_ci": act.interval(loss_crossed[i] - loss_crossed[-1]),
                             "passes": interval[0] > 0}
    calibration = []
    for j, step in enumerate(act.PRIMARY_STEPS):
        limit = act.interval(sampled_model_cdf[:, j] - sampled_truth_cdf[:, j])
        crossed_limit = act.interval(sampled_model_cdf[:, j] - crossed_truth_cdf[:, j])
        calibration.append({"exposure": step, "human_cdf": float(truth_cdf[:, j].mean()),
                            "model_cdf": float(model_cdf[:, j].mean()),
                            "joint_minus_human": float((model_cdf[:, j] - truth_cdf[:, j]).mean()),
                            "paired_material_95_ci": limit, "paired_crossed_95_ci": crossed_limit,
                            "within_margin": limit[0] > -act.CDF_MARGIN and limit[1] < act.CDF_MARGIN})
    return {"trials": int(totals.sum()), "materials": n_items, "readers": n_readers,
            "macro_nll_nats": {name: float(loss[i].mean()) for i, name in enumerate(METHODS)},
            "paired_shared_gain": comparisons, "joint_calibration": calibration,
            "passes_nll": all(comparisons[name]["passes"] for name in ("surface", "qwen_independent")),
            "passes_cdf": all(row["within_margin"] for row in calibration)}


def stage_score(source: Path, archives: Path, forecasts: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    data = read_materials(source, archives)
    seal_file = forecasts / "seal.json"
    expected = {**data.binding, "file_sha256": {name: sha256(forecasts / name) for name in ("models.json", "forecasts.jsonl")}}
    if disjoint.read_json(seal_file) != expected:
        raise ValueError("Expected predictions sealed before evaluation")
    groups = {ACT: data.act_cues.families, CORE: data.core_holdout.families}
    ordered = [(study, family) for study, families in groups.items() for family in families]
    rows = read_jsonl(forecasts / "forecasts.jsonl")
    mass_rows = {key: [] for key in groups}
    for row, (study, family) in zip(rows, ordered, strict=True):
        if (tuple(row) != ("study", "family", "log_masses") or row["study"] != study or row["family"] != family
                or tuple(row["log_masses"]) != METHODS):
            raise ValueError("Forecast materials or methods are out of order")
        mass_rows[study].append([row["log_masses"][method] for method in METHODS])
    masses = {study: np.asarray(value).transpose(1, 0, 2) for study, value in mass_rows.items()}
    for study, value in masses.items():
        if (value.shape != (len(METHODS), len(groups[study]), HORIZON + 1) or not np.isfinite(value).all()
                or not np.allclose(logsumexp(value, axis=-1), 0, rtol=0, atol=1e-12)):
            raise ValueError("Wrong probability space in the sealed forecasts")
    trials = source / "core-vocabulary/raw/data_trials.csv"
    act.checked(trials, TRIAL_TABLE_SHA256)
    core, _ = holdout_reader_counts(trials, data.core_split, list(groups[CORE]))
    german, _ = act.observations(source / "act-german/assessment-v1", data.act_rows)
    labels = np.asarray([data.core_split[f]["condition"] for f in groups[CORE]])
    if core.sum() != 1603 or german.sum() != 1999:
        raise ValueError("Holdout observations do not reproduce frozen trial totals")
    results = {ACT: evaluate(masses[ACT], german, [np.arange(len(groups[ACT]))]),
               CORE: evaluate(masses[CORE], core, [np.flatnonzero(labels == condition) for condition in CONDITIONS])}
    write_json(output, {"forecast_seal_sha256": sha256(seal_file), "results": results,
                        "passes_both_nll": all(row["passes_nll"] for row in results.values()),
                        "passes_both_cdf": all(row["passes_cdf"] for row in results.values())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for name in ("forecast", "score"):
        command = commands.add_parser(name)
        for field in ("source", "archives", "output"):
            command.add_argument(f"--{field}", type=Path, required=True)
        if name == "score":
            command.add_argument("--forecasts", type=Path, required=True)
    args = vars(parser.parse_args())
    (stage_forecast if args.pop("stage") == "forecast" else stage_score)(**args)


if __name__ == "__main__":
    main()
