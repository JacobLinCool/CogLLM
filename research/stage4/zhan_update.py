"""Fit the nested pronoun-update head with the frozen Zhan design."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..stage3.probe import _jsonl
from ..stage3.zhan_landmark_probe import CONDITIONS, load_trials
from .zhan_weak import ARMS, EFFECT_WEIGHT, LABELS_PER_CELL, SEEDS, _contrast_coefficients, _fit_head


N_FOLDS = 5
HIDDEN_WIDTH = 2048


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(data, output):
    families, ids, counts, verbs, _ = load_trials(data)
    by_stimulus = defaultdict(list)
    for row in _jsonl(data / "trials.jsonl"):
        by_stimulus[row["stimulus_id"]].append(row["observations"]["reMention"]["value"] == "np1")
    labels = [np.asarray(by_stimulus[stimulus_id], dtype=np.int64) for stimulus_id in ids]
    positives, totals = counts.reshape(-1, 2).T
    if any(
        len(values) != total or values.sum() != positive or total <= LABELS_PER_CELL
        for values, positive, total in zip(labels, positives, totals, strict=True)
    ):
        raise ValueError("Published trial counts and ordered trial rows disagree")

    contrast = _contrast_coefficients(verbs)
    draws = []
    for seed in SEEDS:
        rng = np.random.default_rng(np.random.SeedSequence([seed, N_FOLDS]))
        direct = np.array([
            rng.choice(values, size=LABELS_PER_CELL, replace=False).sum()
            for values in labels
        ])
        remaining = (positives - direct) / (totals - LABELS_PER_CELL)
        draws.append({
            "seed": seed,
            "limited_positives": direct.tolist(),
            "effect_target": float(contrast @ remaining),
        })
    _write_json(output, {
        "family_ids": families,
        "stimulus_ids": ids,
        "trials_per_limited_stimulus": LABELS_PER_CELL,
        "weak_effect_weight": EFFECT_WEIGHT,
        "source_sha256": {
            name: _sha256(data / name) for name in ("stimuli.jsonl", "trials.jsonl", "splits.json")
        },
        "train_cells": list(range(len(ids))),
        "full_counts": np.column_stack((positives, totals)).tolist(),
        "contrast_coefficients": contrast.tolist(),
        "draws": draws,
    })


def _load_features(torch, path, ids, device):
    with np.load(path, allow_pickle=False) as cache:
        cached_ids = cache["ids"].tolist()
        hidden = cache["hidden"]
    by_id = {stimulus: index for index, stimulus in enumerate(cached_ids)}
    if len(cached_ids) != len(ids) or set(by_id) != set(ids) or hidden.shape != (len(ids), HIDDEN_WIDTH):
        raise ValueError("Expected one cached 2,048-dimensional vector per stimulus")
    hidden = torch.as_tensor(hidden[[by_id[stimulus] for stimulus in ids]], dtype=torch.float64, device=device)
    norms = torch.linalg.vector_norm(hidden, dim=1, keepdim=True)
    if not torch.isfinite(norms).all().item() or (norms <= 0).any().item():
        raise ValueError("Readout input norms must be finite and nonzero")
    return hidden / norms


def update_features(torch, hidden):
    """Append a zero block before the cue and normalized h1 minus h0 after it."""
    if hidden.ndim != 2 or hidden.shape[0] % 2 or hidden.shape[1] != HIDDEN_WIDTH:
        raise ValueError("Expected consecutive (free, pronoun) pairs of normalized states")
    delta = torch.zeros_like(hidden)
    delta[1::2] = hidden[1::2] - hidden[::2]
    return torch.cat((hidden, delta), dim=1)


def _fit_split(torch, features, payload, train_cells):
    draws = payload["draws"]
    if payload["train_cells"] != train_cells.tolist() or tuple(row["seed"] for row in draws) != SEEDS:
        raise ValueError("Unexpected training cells or label draws")

    counts = np.asarray(payload["full_counts"])
    coefficients = np.asarray(payload["contrast_coefficients"])
    if counts.shape != (len(features), 2) or coefficients.shape != (len(features),):
        raise ValueError("Invalid shape for training counts or contrast")
    if (
        not np.issubdtype(counts.dtype, np.integer)
        or (counts[:, 0] < 0).any()
        or (counts[:, 1] <= LABELS_PER_CELL).any()
        or (counts[:, 0] > counts[:, 1]).any()
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("Invalid binomial counts or contrast coefficients")

    positive, total = features.new_tensor(counts).unbind(dim=1)
    contrast = features.new_tensor(coefficients)
    total_count = float(counts[:, 1].sum())
    limited_total = features.new_full((len(features),), LABELS_PER_CELL)
    fit_diagnostics = []
    parameters = np.empty((len(ARMS), len(SEEDS), features.shape[1] + 1))

    def fit_one(arm, seed, positives, totals, target, weight):
        head, diagnostic = _fit_head(
            torch, features, positives, totals, total_count, contrast, target, weight,
        )
        parameters[arm, seed] = head.cpu().numpy()
        fit_diagnostics.append({
            "arm": ARMS[arm], "seed": seed, "effect_target": target if weight else None, **diagnostic,
        })

    fit_one(2, 0, positive, total, 0, 0)
    for draw in draws:
        direct = np.asarray(draw["limited_positives"])
        target = draw["effect_target"]
        if (
            direct.shape != (len(features),)
            or not np.issubdtype(direct.dtype, np.integer)
            or (direct < 0).any()
            or (direct > LABELS_PER_CELL).any()
            or (direct > counts[:, 0]).any()
            or (counts[:, 0] - direct > counts[:, 1] - LABELS_PER_CELL).any()
            or not np.isfinite(target)
        ):
            raise ValueError("Invalid limited binomial counts or weak target")
        remaining = (counts[:, 0] - direct) / (counts[:, 1] - LABELS_PER_CELL)
        if abs(float(coefficients @ remaining) - target) > 1e-12:
            raise ValueError("Weak target disagrees with the remaining training labels")
        direct = features.new_tensor(direct)
        fit_one(0, draw["seed"], direct, limited_total, 0, 0)
        fit_one(1, draw["seed"], direct, limited_total, target, EFFECT_WEIGHT)

    parameters[2, 1:] = parameters[2, 0]
    if not np.isfinite(parameters).all():
        raise RuntimeError("Incomplete or nonfinite fitted readout")
    return parameters, fit_diagnostics


def fit(design_path, full_design_path, features_path, output, device):
    import torch

    start = perf_counter()
    design = json.loads(design_path.read_text(encoding="utf-8"))
    full = json.loads(full_design_path.read_text(encoding="utf-8"))
    families = design["family_ids"]
    ids = design["stimulus_ids"]
    family_folds = np.asarray(design["family_folds"])
    if (
        len(families) != 36
        or len(set(families)) != len(families)
        or ids != [f"{family}-{condition}" for family in families for condition in CONDITIONS]
        or family_folds.shape != (len(families),)
        or set(family_folds) != set(range(N_FOLDS))
        or [row["fold"] for row in design["fold_designs"]] != list(range(N_FOLDS))
        or design["trials_per_limited_stimulus"] != LABELS_PER_CELL
        or design["weak_effect_weight"] != EFFECT_WEIGHT
        or any(full[key] != design[key] for key in (
            "family_ids", "stimulus_ids", "source_sha256", "trials_per_limited_stimulus", "weak_effect_weight",
        ))
    ):
        raise ValueError("Training payloads disagree with the frozen Zhan design")
    if device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.get_device_name().endswith(" L4"):
            raise RuntimeError("Requested training on a Colab L4")
        torch.cuda.reset_peak_memory_stats()

    hidden = _load_features(torch, features_path, ids, device)
    expanded = update_features(torch, hidden)
    width = expanded.shape[1] + 1
    predictions = np.full((len(ARMS), len(SEEDS), len(ids)), np.nan)
    parameters = np.empty((len(ARMS), len(SEEDS), N_FOLDS, width))
    diagnostics = []
    folds = np.repeat(family_folds, 2)
    for payload in design["fold_designs"]:
        fold = payload["fold"]
        train = folds != fold
        mask = torch.as_tensor(train, device=device)
        head, report = _fit_split(torch, expanded[mask], payload, np.flatnonzero(train))
        parameters[:, :, fold] = head
        head = expanded.new_tensor(head)
        logits = head[..., :-1] @ expanded[~mask].T + head[..., -1:]
        predictions[:, :, ~train] = logits.sigmoid().cpu().numpy()
        diagnostics.extend({"scope": "zhan_oof", "head": "update", "fold": fold, **row} for row in report)
        print(f"Fitted pronoun-update fold {fold + 1}/{N_FOLDS}", flush=True)

    if not np.isfinite(predictions).all():
        raise RuntimeError("Incomplete or nonfinite out-of-fold probabilities")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"arms": np.array(ARMS), "seeds": np.array(SEEDS)}
    np.savez_compressed(
        output / "heads.npz", family_ids=np.array(families), parameters=parameters,
        predictions=predictions.reshape(len(ARMS), len(SEEDS), len(families), 2), **metadata,
    )

    external = {}
    cells = np.arange(len(ids))
    for variant, features in (("baseline", hidden), ("update", expanded)):
        external[f"{variant}_parameters"], report = _fit_split(torch, features, full, cells)
        diagnostics.extend({"scope": "all_zhan_for_external", "head": variant, **row} for row in report)
        print(f"Fitted all-Zhan {variant} head", flush=True)
    np.savez_compressed(output / "external_heads.npz", **external, **metadata)
    _write_json(output / "fitting.json", {
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "dtype": "float64",
        "train_s": perf_counter() - start,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "design_sha256": _sha256(design_path),
        "full_design_sha256": _sha256(full_design_path),
        "features_sha256": _sha256(features_path),
        "checkpoints_sha256": {name: _sha256(output / name) for name in ("heads.npz", "external_heads.npz")},
        "max_gradient_absolute": max(row["gradient_max_absolute"] for row in diagnostics),
        "fits": diagnostics,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--data", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("fit")
    run.add_argument("--design", type=Path, required=True)
    run.add_argument("--full-design", type=Path, required=True)
    run.add_argument("--features", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=("cuda", "cpu"), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data, args.output)
    else:
        fit(args.design, args.full_design, args.features, args.output, args.device)
