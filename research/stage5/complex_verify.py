"""Independently replay sealed CompLex predictions and audit the fitted heads."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("target_only", "target_end", "full_sentence")
PERCENTAGES = (25, 50, 100)
SEEDS = (0, 1, 2, 3, 4)
SPLIT_SIZES = {"train": 8_667, "trial": 503, "test": 1_094}
LAYER_WIDTH = 2_048
MAX_REPLAY_ERROR = 1e-10
MAX_STATIONARITY_ERROR = 1e-8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_file(path: Path, expected: dict) -> None:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    require(path.stat().st_size == expected["bytes"], f"File size mismatch: {path}")
    require(digest == expected["sha256"], f"SHA-256 mismatch: {path}")


def split_rows(design: Path) -> dict[str, list[dict]]:
    splits = {name: [] for name in SPLIT_SIZES}
    groups = {name: set() for name in SPLIT_SIZES}
    ids = set()
    with design.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            split, identifier, group = row["split"], row["id"], row["context_group"]
            require(split in splits, f"Unexpected split: {split}")
            require(isinstance(identifier, str) and bool(identifier), "Missing observation ID")
            require(isinstance(group, str) and bool(group), "Missing context group")
            require(identifier not in ids, f"Duplicate observation ID: {identifier}")
            require(split != "test" or row["value"] is None, "Test ratings must be absent")
            splits[split].append(row)
            groups[split].add(group)
            ids.add(identifier)
    require({name: len(rows) for name, rows in splits.items()} == SPLIT_SIZES, "Incorrect split sizes")
    require(
        sum(len(groups_in_split) for groups_in_split in groups.values()) == len(set.union(*groups.values())),
        "Canonical context groups overlap between splits",
    )
    return {name: sorted(rows, key=lambda row: row["id"]) for name, rows in splits.items()}


def expected_draws(train_rows: list[dict]) -> tuple[np.ndarray, list[int]]:
    group_order = np.asarray(sorted({row["context_group"] for row in train_rows}))
    group_counts = [math.ceil(Fraction(pct * group_order.size, 100)) for pct in PERCENTAGES]
    masks = np.empty((len(PERCENTAGES), len(SEEDS), len(train_rows)), dtype=np.bool_)
    for seed_idx, seed in enumerate(SEEDS):
        shuffled = np.random.default_rng(seed).permutation(group_order)
        for pct_idx, count in enumerate(group_counts):
            selected = set(shuffled[:count].tolist())
            masks[pct_idx, seed_idx] = [row["context_group"] in selected for row in train_rows]
    return masks, group_counts


def read_block(
    archive: np.lib.npyio.NpzFile, key: str, row_positions: np.ndarray
) -> np.ndarray:
    bank = archive[key]
    require(
        bank.dtype == np.float32 and bank.ndim == 2 and bank.shape[1] == LAYER_WIDTH,
        f"Invalid saved hidden-state block: {key}",
    )
    if key == "target_end":
        require(len(bank) == sum(SPLIT_SIZES.values()), "Incorrect target_end row count")
        index = row_positions
    else:
        index_map = archive[f"{key}_index"]
        require(
            index_map.dtype == np.int64
            and index_map.shape == (sum(SPLIT_SIZES.values()),)
            and index_map.min() >= 0
            and index_map.max() < len(bank),
            f"Invalid saved feature mapping: {key}",
        )
        index = index_map[row_positions]
    matrix = bank[index].astype(np.float64)
    norm = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))
    require(np.all(np.isfinite(norm) & (norm > 0)), f"Nonfinite or zero-norm feature: {key}")
    matrix /= norm[:, None]
    return matrix


def verify(design: Path, features: Path, fit_dir: Path) -> dict:
    metadata = json.loads((fit_dir / "fitting.json").read_text(encoding="utf-8"))
    check_file(design, metadata["inputs"]["design"])
    check_file(features, metadata["inputs"]["features"])
    for name in ("heads.npz", "predictions.npz"):
        check_file(fit_dir / name, metadata["artifacts"][name])

    splits = split_rows(design)
    train_rows, test_rows = splits["train"], splits["test"]
    train_ids = [row["id"] for row in train_rows]
    test_ids = [row["id"] for row in test_rows]
    ratings = np.asarray([row["value"] for row in train_rows], dtype=np.float64)
    require(np.all((ratings >= 0) & (ratings <= 1)), "Invalid training ratings")
    design_ids = {row["id"] for rows in splits.values() for row in rows}
    masks, expected_group_counts = expected_draws(train_rows)
    metadata_axes = {"methods": METHODS, "fractions": PERCENTAGES, "seeds": SEEDS}
    parameter_shape = (len(METHODS), len(PERCENTAGES), len(SEEDS))
    max_error = 0.0
    max_stationarity = 0.0
    fits_checked = 0
    grid = {}
    for fit in metadata["grid"]:
        key = (fit["method"], fit["fraction"], tuple(fit["seeds"]))
        require(key not in grid, f"Duplicate fit log record: {key}")
        grid[key] = fit
    with ExitStack() as stack:
        feature_data = stack.enter_context(np.load(features, allow_pickle=False))
        heads = stack.enter_context(np.load(fit_dir / "heads.npz", allow_pickle=False))
        sealed = stack.enter_context(np.load(fit_dir / "predictions.npz", allow_pickle=False))
        for axis, expected in metadata_axes.items():
            require(tuple(metadata[axis]) == expected, f"Incorrect JSON axis: {axis}")
            require(tuple(heads[axis].tolist()) == expected, f"Incorrect heads axis: {axis}")
            require(tuple(sealed[axis].tolist()) == expected, f"Incorrect predictions axis: {axis}")
        require(heads["train_ids"].tolist() == train_ids, "Train head IDs differ from design")
        require(sealed["ids"].tolist() == test_ids, "Sealed test IDs differ from design")
        actual_masks = heads["train_masks"]
        require(actual_masks.dtype == np.bool_ and np.array_equal(masks, actual_masks), "Train masks differ from expected draws")
        predictions = sealed["predictions"]
        intercepts = heads["intercepts"]
        require(
            predictions.dtype == np.float64 and predictions.shape == (*parameter_shape, len(test_ids)),
            "Incorrect held-out prediction array",
        )
        require(np.all((predictions >= 0) & (predictions <= 1)), "Predictions outside [0, 1]")
        require(
            intercepts.dtype == np.float64 and intercepts.shape == parameter_shape and np.isfinite(intercepts).all(),
            "Incorrect intercept array",
        )
        require(np.all(intercepts[:, -1] == intercepts[:, -1, :1]), "Full-data intercept differs by seed")
        require(np.all(predictions[:, -1] == predictions[:, -1, :1]), "Full-data prediction differs by seed")
        feature_ids = feature_data["ids"]
        require(feature_ids.dtype.kind == "U" and feature_ids.ndim == 1, "Invalid feature IDs")
        feature_positions = {identifier: index for index, identifier in enumerate(feature_ids.tolist())}
        require(
            len(feature_positions) == len(feature_ids) and feature_positions.keys() == design_ids,
            "Feature IDs differ from design",
        )
        # Arrange the joined features as one training block followed by held-out rows.
        positions = np.asarray([feature_positions[identifier] for identifier in train_ids + test_ids])
        for method_idx, method in enumerate(METHODS):
            weights = heads[f"weights_{method}"]
            width = 2 * LAYER_WIDTH if method == "full_sentence" else LAYER_WIDTH
            require(
                weights.dtype == np.float64 and weights.shape == (*parameter_shape[1:], width)
                and np.isfinite(weights).all(),
                f"Incorrect weight array: {method}",
            )
            require(np.all(weights[-1] == weights[-1, :1]), f"Full-data weights vary by seed: {method}")
            if method == "full_sentence":
                blocks = (read_block(feature_data, name, positions) for name in ("target_end", "sentence_end"))
                matrix = np.concatenate(tuple(blocks), axis=1)
                matrix /= math.sqrt(2)
            else:
                matrix = read_block(feature_data, method, positions)
            train, test = matrix[: len(train_ids)], matrix[len(train_ids) :]
            raw = weights @ test.T + intercepts[method_idx, :, :, None]
            require(np.isfinite(raw).all(), f"Nonfinite replayed estimates: {method}")
            max_error = max(max_error, float(np.max(np.abs(np.clip(raw, 0, 1) - predictions[method_idx]))))
            for pct_idx, pct in enumerate(PERCENTAGES):
                for seed_idx, seed in enumerate(SEEDS if pct < 100 else SEEDS[:1]):
                    mask = masks[pct_idx, seed_idx]
                    key = (method, pct, (seed,) if pct < 100 else SEEDS)
                    fit = grid.pop(key)
                    penalty = mask.sum().item() / len(train_ids)
                    require(fit["train_rows"] == mask.sum(), f"Incorrect logged row count: {key}")
                    require(fit["train_context_groups"] == expected_group_counts[pct_idx], f"Incorrect logged group count: {key}")
                    require(fit["l2_penalty"] == penalty, f"Incorrect logged penalty: {key}")
                    x = train[mask]
                    y = ratings[mask]
                    w = weights[pct_idx, seed_idx]
                    error = x @ w + intercepts[method_idx, pct_idx, seed_idx] - y
                    weight_gradient = x.T @ error + penalty * w
                    intercept_gradient = abs(error.sum())
                    scale = max(1.0, np.linalg.norm(x.T @ (y - y.mean())))
                    stationarity = max(np.linalg.norm(weight_gradient), intercept_gradient) / scale
                    require(np.isfinite(stationarity), f"Nonfinite ridge stationarity error: {key}")
                    max_stationarity = max(max_stationarity, float(stationarity))
                    fits_checked += 1
        require(not grid and fits_checked == metadata["unique_fits"] == 33, "Incorrect unique fit inventory")
        require(max_error <= MAX_REPLAY_ERROR, f"Held-out replay error {max_error} exceeds {MAX_REPLAY_ERROR}")
        require(
            max_stationarity <= MAX_STATIONARITY_ERROR,
            f"Ridge stationarity error {max_stationarity} exceeds {MAX_STATIONARITY_ERROR}",
        )
    return {
        "unique_heads_checked": fits_checked,
        "predictions_replayed": int(np.prod(parameter_shape) * len(test_ids)),
        "split_groups": {name: len({row["context_group"] for row in rows}) for name, rows in splits.items()},
        "train_groups_by_percentage": dict(zip(PERCENTAGES, expected_group_counts, strict=True)),
        "train_rows_by_percentage_and_seed": masks.sum(axis=-1).tolist(),
        "max_absolute_clip_replay_error": max_error,
        "max_relative_ridge_stationarity_error": max_stationarity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--fit-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.design, args.features, args.fit_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
