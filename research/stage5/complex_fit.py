"""Fit fixed-penalty lexical rating probes to frozen Qwen3.5-2B features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


METHODS = ("target_only", "target_end", "full_sentence")
FRACTIONS = (25, 50, 100)
SEEDS = (0, 1, 2, 3, 4)
EXPECTED_COUNTS = {"train": 8_667, "trial": 503, "test": 1_094}
HIDDEN_SIZE = 2_048


def load_design(path: Path) -> list[dict]:
    rows = sorted(
        (json.loads(line) for line in path.read_text().splitlines() if line),
        key=lambda row: row["id"],
    )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Observation IDs must be unique")
    groups: dict[str, str] = {}
    counts = dict.fromkeys(EXPECTED_COUNTS, 0)
    for row in rows:
        split, group, value = row["split"], row["context_group"], row["value"]
        if split not in counts or not isinstance(group, str) or not group:
            raise ValueError(f"Invalid split or context group for {row['id']}")
        if groups.setdefault(group, split) != split:
            raise ValueError(f"Context group appears in multiple splits: {group}")
        if split == "test":
            if value is not None:
                raise ValueError("Test ratings must be withheld from the fitting runtime")
        elif type(value) not in (float, int) or not 0 <= value <= 1:
            raise ValueError(f"Invalid training or trial rating for {row['id']}")
        counts[split] += 1
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected split counts: {counts}; expected {EXPECTED_COUNTS}")
    return rows


def training_masks(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, group_index = np.unique(groups, return_inverse=True)
    masks = np.empty((len(FRACTIONS), len(SEEDS), len(groups)), dtype=bool)
    counts = np.asarray([math.ceil(len(unique) * fraction / 100) for fraction in FRACTIONS])
    for seed_index, seed in enumerate(SEEDS):
        permutation = np.random.default_rng(seed).permutation(len(unique))
        rank = np.argsort(permutation)[group_index]
        masks[:, seed_index] = rank < counts[:, None]
    return masks, counts


def load_features(path: Path, ids: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    with np.load(path, allow_pickle=False) as archive:
        feature_ids = archive["ids"]
        if feature_ids.ndim != 1 or feature_ids.dtype.kind != "U":
            raise ValueError("Feature IDs must be a one-dimensional Unicode array")
        lookup = {identifier: index for index, identifier in enumerate(feature_ids.tolist())}
        if len(lookup) != len(feature_ids) or set(lookup) != set(ids):
            raise ValueError("Features must cover exactly the design observation IDs")
        order = np.asarray([lookup[identifier] for identifier in ids])
        for name in ("target_only", "target_end", "sentence_end"):
            matrix = archive[name]
            if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[1] != HIDDEN_SIZE:
                raise ValueError(f"Invalid float32 hidden-state matrix: {name}")
            tensor = torch.as_tensor(matrix, dtype=torch.float64, device=device)
            norms = torch.linalg.vector_norm(tensor, dim=1, keepdim=True)
            if not torch.logical_and(torch.isfinite(norms), norms > 0).all().item():
                raise ValueError(f"Non-finite or zero-norm feature: {name}")
            tensor.div_(norms)
            if name == "target_end":
                if len(matrix) != len(ids):
                    raise ValueError("Target-end features require one row per observation")
                indices = order
            else:
                index_map = archive[f"{name}_index"]
                if (
                    index_map.dtype != np.int64
                    or index_map.shape != (len(ids),)
                    or index_map.min() < 0
                    or index_map.max() >= len(matrix)
                ):
                    raise ValueError(f"Invalid deduplicated feature index: {name}")
                indices = index_map[order]
            features[name] = tensor.index_select(0, torch.as_tensor(indices, device=device))
    return features


def fit_ridge(
    features: torch.Tensor, values: torch.Tensor, penalty: float
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Minimize sum squared error plus penalty times squared weight norm."""
    feature_mean = features.mean(dim=0)
    value_mean = values.mean()
    centered = features - feature_mean
    centered_values = values - value_mean
    rhs = centered.T @ centered_values
    if centered.shape[0] >= centered.shape[1]:
        gram = centered.T @ centered
        gram.diagonal().add_(penalty)
        weight = torch.cholesky_solve(rhs[:, None], torch.linalg.cholesky(gram)).squeeze(1)
    else:
        gram = centered @ centered.T
        gram.diagonal().add_(penalty)
        dual = torch.cholesky_solve(centered_values[:, None], torch.linalg.cholesky(gram))
        weight = (centered.T @ dual).squeeze(1)
    intercept = value_mean - feature_mean @ weight
    gradient = centered.T @ (centered @ weight - centered_values) + penalty * weight
    relative_residual = (gradient.norm() / rhs.norm().clamp_min(1e-12)).item()
    if not math.isfinite(relative_residual) or relative_residual > 1e-8:
        raise ArithmeticError(f"Ridge normal equation residual too large: {relative_residual}")
    return weight, intercept, relative_residual


def file_info(path: Path) -> dict:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size}


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    pending = path.with_suffix(".partial.npz")
    np.savez_compressed(pending, **arrays)
    pending.replace(path)


def run(design: Path, archive: Path, output: Path, device: torch.device) -> None:
    started = time.perf_counter()
    if device.type == "cuda":
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(device) != "NVIDIA L4":
            raise RuntimeError("CUDA fitting requires exactly one NVIDIA L4")
        torch.cuda.reset_peak_memory_stats(device)
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    rows = load_design(design)
    ids = np.asarray([row["id"] for row in rows])
    split = np.asarray([row["split"] for row in rows])
    train_ids, trial_ids, test_ids = (ids[split == name] for name in EXPECTED_COUNTS)
    groups = np.asarray([row["context_group"] for row in rows if row["split"] == "train"])
    masks, group_counts = training_masks(groups)
    train_values, trial_values = (
        torch.tensor(
            [row["value"] for row in rows if row["split"] == name],
            dtype=torch.float64,
            device=device,
        )
        for name in ("train", "trial")
    )
    splits = {
        name: torch.as_tensor(np.flatnonzero(split == name), device=device)
        for name in EXPECTED_COUNTS
    }
    features = load_features(archive, ids.tolist(), device)
    method_features = {
        "target_only": features["target_only"],
        "target_end": features["target_end"],
        "full_sentence": torch.cat(
            (features["target_end"], features["sentence_end"]), dim=1
        ).mul_(1 / math.sqrt(2)),
    }
    del features
    grid_shape = (len(METHODS), len(FRACTIONS), len(SEEDS))
    predictions = np.empty((*grid_shape, len(test_ids)), dtype=np.float64)
    intercepts = np.empty(grid_shape, dtype=np.float64)
    weights = {
        method: np.empty((*grid_shape[1:], method_features[method].shape[1]), dtype=np.float64)
        for method in METHODS
    }
    grid = []
    fit_started = time.perf_counter()
    for method_index, method in enumerate(METHODS):
        method_matrix = method_features.pop(method)
        train, trial, test = (method_matrix.index_select(0, splits[name]) for name in EXPECTED_COUNTS)
        del method_matrix
        for fraction_index, fraction in enumerate(FRACTIONS):
            seed_indices = range(len(SEEDS)) if fraction < 100 else range(1)
            for seed_index in seed_indices:
                sample = np.flatnonzero(masks[fraction_index, seed_index])
                indices = torch.as_tensor(sample, device=device)
                penalty = len(sample) / len(train_ids)
                w, b, residual = fit_ridge(
                    train.index_select(0, indices), train_values.index_select(0, indices), penalty
                )
                test_prediction = (test @ w + b).clamp_(0, 1).cpu().numpy()
                trial_mae = ((trial @ w + b).clamp_(0, 1) - trial_values).abs().mean().item()
                seed_slot = seed_index if fraction < 100 else slice(None)
                predictions[method_index, fraction_index, seed_slot] = test_prediction
                intercepts[method_index, fraction_index, seed_slot] = b.item()
                weights[method][fraction_index, seed_slot] = w.cpu().numpy()
                grid.append({
                    "method": method,
                    "fraction": fraction,
                    "seeds": [SEEDS[seed_index]] if fraction < 100 else list(SEEDS),
                    "train_rows": len(sample),
                    "train_context_groups": int(group_counts[fraction_index]),
                    "l2_penalty": penalty,
                    "normal_equation_relative_residual": residual,
                    "trial_mae": trial_mae,
                })
        print(json.dumps({
            "fitted_method": method,
            "elapsed_seconds": time.perf_counter() - fit_started,
        }), flush=True)
    fit_seconds = time.perf_counter() - fit_started
    if not np.isfinite(predictions).all():
        raise ArithmeticError("Non-finite held-out predictions")

    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "methods": np.asarray(METHODS),
        "fractions": np.asarray(FRACTIONS),
        "seeds": np.asarray(SEEDS),
    }
    save_npz(output / "predictions.npz", **metadata, ids=test_ids, predictions=predictions)
    save_npz(
        output / "heads.npz",
        **metadata,
        train_ids=train_ids,
        train_masks=masks,
        intercepts=intercepts,
        **{f"weights_{method}": matrix for method, matrix in weights.items()},
    )
    report = {
        "hardware": hardware,
        "fitting_dtype": "float64",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "methods": list(METHODS),
        "fractions": list(FRACTIONS),
        "seeds": list(SEEDS),
        "split_rows": {"train": len(train_ids), "trial": len(trial_ids), "test": len(test_ids)},
        "split_context_groups": {
            name: len({row["context_group"] for row in rows if row["split"] == name})
            for name in EXPECTED_COUNTS
        },
        "feature_normalization": (
            "L2-normalize each hidden-state row. Concatenate the normalized target_end and "
            "sentence_end rows for full_sentence and divide by sqrt(2)."
        ),
        "training_sampling": (
            "Select nested prefixes of the permuted canonical context groups for each seed. "
            "Round each group count up. Share the full-data fit across seeds."
        ),
        "fitting_objective": (
            "Sum squared error plus (n / N_all_train) times squared L2 weight norm "
            "with an unpenalized intercept and a fixed penalty schedule."
        ),
        "prediction_bounds": [0, 1],
        "trial_diagnostics": "Diagnostic-only MAE after clipping estimates to prediction_bounds.",
        "unique_fits": len(grid),
        "fit_seconds": fit_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "inputs": {"design": file_info(design), "features": file_info(archive)},
        "artifacts": {name: file_info(output / name) for name in ("predictions.npz", "heads.npz")},
        "grid": grid,
    }
    report_path = output / "fitting.json"
    pending = report_path.with_suffix(".partial.json")
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pending.replace(report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run(args.design, args.features, args.output, torch.device(args.device))


if __name__ == "__main__":
    main()
