"""Prepare fold-local Zhan supervision and fit a readout on cached causal states."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..stage3.zhan_landmark_probe import VERB_TYPES, load_trials
from ..stage3.probe import _jsonl


LABELS_PER_CELL = 4
SEEDS = tuple(range(12))
ARMS = ("limited", "weak", "full")
EFFECT_WEIGHT = 1.0
GRADIENT_LIMIT = 5e-7


def _contrast_coefficients(verbs):
    first = verbs == VERB_TYPES[0]
    second = verbs == VERB_TYPES[1]
    if not first.any() or not second.any() or not (first | second).all():
        raise ValueError("Both expected verb types must appear in a training fold")
    weights = second / second.sum() - first / first.sum()
    return np.column_stack((-weights, weights)).ravel()


def prepare(data, output):
    families, stimulus_ids, counts, verbs, folds = load_trials(data)
    by_stimulus = defaultdict(list)
    for row in _jsonl(data / "trials.jsonl"):
        by_stimulus[row["stimulus_id"]].append(row["observations"]["reMention"]["value"] == "np1")
    labels = [np.array(by_stimulus[stimulus_id], dtype=np.int64) for stimulus_id in stimulus_ids]
    full_positive, full_total = (counts[..., index].ravel() for index in (0, 1))
    if any(
        len(values) != full_total[index] or values.sum() != full_positive[index] or len(values) <= LABELS_PER_CELL
        for index, values in enumerate(labels)
    ):
        raise ValueError("Published trial counts and ordered trial rows disagree")

    fold_designs = []
    for fold in range(5):
        train = folds != fold
        train_cells = np.flatnonzero(np.repeat(train, 2))
        totals = full_total[train_cells]
        positives = full_positive[train_cells]
        contrast = _contrast_coefficients(verbs[train])
        draws = []
        for seed in SEEDS:
            rng = np.random.default_rng(np.random.SeedSequence([seed, fold]))
            direct = np.array([
                rng.choice(labels[index], size=LABELS_PER_CELL, replace=False).sum()
                for index in train_cells
            ])
            remaining = (positives - direct) / (totals - LABELS_PER_CELL)
            draws.append({
                "seed": seed,
                "limited_positives": direct.tolist(),
                "effect_target": float(contrast @ remaining),
            })
        fold_designs.append({
            "fold": fold,
            "train_cells": train_cells.tolist(),
            "full_counts": np.column_stack((positives, totals)).tolist(),
            "contrast_coefficients": contrast.tolist(),
            "draws": draws,
        })

    design = {
        "family_ids": families,
        "stimulus_ids": stimulus_ids,
        "family_folds": folds.tolist(),
        "trials_per_limited_stimulus": LABELS_PER_CELL,
        "weak_effect_weight": EFFECT_WEIGHT,
        "source_sha256": {
            name: hashlib.sha256((data / name).read_bytes()).hexdigest()
            for name in ("stimuli.jsonl", "trials.jsonl", "splits.json")
        },
        "fold_designs": fold_designs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(design, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _fit_head(torch, features, positives, totals, ridge_denominator, contrast, effect_target, effect_weight):
    weights = features.new_zeros(features.shape[1] + 1, requires_grad=True)
    opt = torch.optim.LBFGS(
        [weights], max_iter=250, max_eval=300, tolerance_grad=1e-9,
        tolerance_change=1e-14, line_search_fn="strong_wolfe",
    )
    direct_count = totals.sum()

    def objective():
        logits = features @ weights[:-1] + weights[-1]
        direct = (totals * torch.nn.functional.softplus(logits) - positives * logits).sum() / direct_count
        ridge = weights[:-1].square().sum() / (2 * ridge_denominator)
        effect = contrast @ logits.sigmoid()
        loss = direct + ridge + (effect_weight / 2) * (effect - effect_target).square()
        return loss, direct, effect

    def closure():
        opt.zero_grad()
        loss, _, _ = objective()
        loss.backward()
        return loss

    opt.step(closure)
    opt.zero_grad()
    loss, nll, effect = objective()
    gradient, = torch.autograd.grad(loss, weights)
    residual = float(gradient.detach().abs().max().item())
    optimizer_state = opt.state[weights]
    if not np.isfinite(residual) or residual > GRADIENT_LIMIT:
        raise RuntimeError(f"Readout optimizer did not converge: maximum gradient {residual:.3g}")
    return weights.detach(), {
        "objective": loss.item(),
        "direct_nll": nll.item(),
        "training_effect": effect.item(),
        "gradient_max_absolute": residual,
        "function_evaluations": optimizer_state["func_evals"],
        "optimizer_iterations": optimizer_state["n_iter"],
    }


def fit(design_path, features_path, output, device):
    import torch

    start = perf_counter()
    design = json.loads(design_path.read_text(encoding="utf-8"))
    ids = design["stimulus_ids"]
    family_ids = design["family_ids"]
    folds = np.array(design["family_folds"])
    if (
        len(family_ids) != 36 or len(ids) != 2 * len(family_ids) or len(set(ids)) != len(ids)
        or folds.shape != (len(family_ids),) or set(folds) != set(range(5))
        or design["trials_per_limited_stimulus"] != LABELS_PER_CELL or design["weak_effect_weight"] != EFFECT_WEIGHT
        or [row["fold"] for row in design["fold_designs"]] != list(range(5))
    ):
        raise ValueError("Training payload differs from the fixed design")
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.get_device_name().endswith(" L4")):
        raise RuntimeError("Requested training on a Colab L4")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with np.load(features_path, allow_pickle=False) as source:
        cached_ids = source["ids"].tolist()
        hidden = source["hidden"]
    index_by_id = {stimulus: index for index, stimulus in enumerate(cached_ids)}
    if set(index_by_id) != set(ids) or len(cached_ids) != len(ids) or hidden.shape != (len(ids), 2048):
        raise ValueError("Expected one cached 2,048-dimensional vector per condition")
    features = torch.as_tensor(
        hidden[[index_by_id[stimulus] for stimulus in ids]], dtype=torch.float64, device=device,
    )
    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    if not torch.isfinite(norms).all().item() or (norms <= 0).any().item():
        raise ValueError("Readout input norms must be finite and nonzero")
    features = features / norms

    predictions = np.full((len(ARMS), len(SEEDS), len(ids)), np.nan)
    parameters = np.empty((len(ARMS), len(SEEDS), 5, features.shape[1] + 1))
    diagnostics = []
    for payload in design["fold_designs"]:
        fold = payload["fold"]
        train = np.repeat(folds != fold, 2)
        if payload["train_cells"] != np.flatnonzero(train).tolist() or [row["seed"] for row in payload["draws"]] != list(SEEDS):
            raise ValueError(f"A fitting payload contains unexpected rows for fold {fold}")
        train_indices = torch.as_tensor(train, device=device)
        x_train, x_test = features[train_indices], features[~train_indices]
        full_counts = features.new_tensor(payload["full_counts"])
        contrast = features.new_tensor(payload["contrast_coefficients"])
        pos, totals = full_counts.unbind(dim=1)
        full_count = totals.sum().item()
        limited_totals = features.new_full((len(x_train),), LABELS_PER_CELL)
        if len(pos) != len(x_train) or contrast.shape != pos.shape:
            raise ValueError(f"Invalid fit counts or contrast for fold {fold}")

        def record(arm, seed, head, report):
            parameters[arm, seed, fold] = head.cpu().numpy()
            predictions[arm, seed, ~train] = (x_test @ head[:-1] + head[-1]).sigmoid().cpu().numpy()
            diagnostics.append({"fold": fold, "arm": ARMS[arm], "seed": seed, **report})

        head, report = _fit_head(torch, x_train, pos, totals, full_count, contrast, 0, 0)
        record(2, 0, head, report)
        for draw in payload["draws"]:
            seed = draw["seed"]
            limited_positive = features.new_tensor(draw["limited_positives"])
            for arm, target, weight in ((0, 0, 0), (1, draw["effect_target"], EFFECT_WEIGHT)):
                head, report = _fit_head(
                    torch, x_train, limited_positive, limited_totals, full_count, contrast, target, weight,
                )
                record(arm, seed, head, report)
        print(f"Fitted fold {fold + 1}/5", flush=True)

    predictions[2, 1:] = predictions[2, 0]
    parameters[2, 1:] = parameters[2, 0]
    if not np.isfinite(predictions).all() or not np.isfinite(parameters).all():
        raise RuntimeError("Incomplete or nonfinite fitted readout")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "heads.npz", arms=np.array(ARMS), family_ids=np.array(family_ids),
        predictions=predictions.reshape(len(ARMS), len(SEEDS), len(family_ids), 2),
        parameters=parameters, seeds=np.array(SEEDS),
    )
    report = {
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "dtype": "float64",
        "train_s": perf_counter() - start,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "design_sha256": hashlib.sha256(design_path.read_bytes()).hexdigest(),
        "features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "max_gradient_absolute": max(row["gradient_max_absolute"] for row in diagnostics),
        "fits": diagnostics,
    }
    (output / "fitting.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subcommands.add_parser("prepare")
    prepare_parser.add_argument("--data", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    fit_parser = subcommands.add_parser("fit")
    fit_parser.add_argument("--design", type=Path, required=True)
    fit_parser.add_argument("--features", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data, args.output)
    else:
        fit(args.design, args.features, args.output, args.device)
