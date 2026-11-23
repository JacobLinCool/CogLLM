"""Fit fixed-fold four-option OneStopQA controls on frozen prompt states."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np


METHODS = ("display_position_prior", "question_options", "passage_question_options")
RIDGE = 0.002


def align_features(ids, feature_ids, hidden):
    positions = {key: i for i, key in enumerate(feature_ids)}
    if len(positions) != len(feature_ids) or hidden.shape != (len(positions), 2048) or not set(ids).issubset(positions):
        raise ValueError("OneStop feature inventory does not match visible prompt IDs")
    values = hidden[[positions[key] for key in ids]].astype(np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("OneStop feature norms must be finite and positive")
    return values / norms


def row_losses(probability, counts, correct):
    if probability.shape != counts.shape or not np.isfinite(probability).all() or (probability <= 0).any():
        raise ValueError("Expected complete, positive OneStop choice probabilities")
    total = counts.sum(axis=1)
    gold_probability = probability[np.arange(len(correct)), correct]
    gold_count = counts[np.arange(len(correct)), correct]
    return {
        "choice_nll": -(counts * np.log(probability)).sum(axis=1),
        "choice_brier": total * np.square(probability).sum(axis=1) - 2 * (counts * probability).sum(axis=1) + total,
        "correctness_brier": gold_count * (1 - gold_probability)**2 + (total - gold_count) * gold_probability**2,
    }


def fit_weights(torch, features, counts, train, fold, device):
    values = torch.as_tensor(features, dtype=torch.float64, device=device)
    y = torch.as_tensor(counts, dtype=torch.float64, device=device)
    mask = torch.as_tensor(train, device=device)
    x_train, y_train = values[mask], y[mask]
    total = y_train.sum()
    weights = values.new_zeros((values.shape[1] + 1, 4), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights], max_iter=150, max_eval=180, tolerance_grad=2e-8,
        tolerance_change=1e-13, line_search_fn="strong_wolfe",
    )

    def loss():
        logits = x_train @ weights[:-1] + weights[-1]
        return -(y_train * logits.log_softmax(dim=-1)).sum() / total + RIDGE / 2 * weights[:-1].square().sum()

    def closure():
        optimizer.zero_grad()
        value = loss()
        value.backward()
        return value

    optimizer.step(closure)
    value = loss()
    gradient, = torch.autograd.grad(value, weights)
    residual = float(gradient.abs().max().item())
    if not np.isfinite(residual) or residual > 1e-6:
        raise RuntimeError(f"OneStop readout did not converge for fold {fold}: max gradient {residual:.3g}")
    return weights.detach(), {"loss": float(value.item()), "max_gradient": residual}


def fit_fold(torch, features, counts, train, fold, device):
    weights, diagnostics = fit_weights(torch, features, counts, train, fold, device)
    test = torch.as_tensor(features[~train], dtype=torch.float64, device=device)
    with torch.no_grad():
        p = (test @ weights[:-1] + weights[-1]).softmax(dim=-1).cpu().numpy()
    return p, diagnostics


def evaluate(design_path, features_path, output, device):
    import torch

    started = perf_counter()
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.get_device_name().endswith(" L4")):
        raise RuntimeError("Expected a Colab NVIDIA L4 for baseline fitting")
    torch.set_float32_matmul_precision("highest")
    with np.load(design_path, allow_pickle=False) as design:
        ids = design["onestop_ids"].tolist()
        question_only = design["onestop_question_only_ids"].tolist()
        folds = design["onestop_folds"]
        counts = design["onestop_counts"]
        correct = design["onestop_correct"]
        families = design["onestop_families"]
    if (
        len(ids) != 1872 or counts.shape != (len(ids), 4) or counts.sum() != 1940
        or len(question_only) != len(ids) or correct.shape != (len(ids),)
        or (counts < 0).any() or (correct < 0).any() or (correct > 3).any() or set(folds) != set(range(5))
    ):
        raise ValueError("OneStop inputs do not match the fixed pilot inventory")
    with np.load(features_path, allow_pickle=False) as data:
        feature_ids = data["ids"].tolist()
        hidden = data["hidden"]
        blocks = {
            METHODS[1]: align_features(question_only, feature_ids, hidden),
            METHODS[2]: align_features(ids, feature_ids, hidden),
        }
    predictions = np.full((len(METHODS), len(ids), 4), np.nan)
    diagnostics = []
    for fold in range(5):
        train = folds != fold
        prevalence = counts[train].sum(axis=0)
        predictions[0, ~train] = prevalence / prevalence.sum()
        for method, features in blocks.items():
            values, record = fit_fold(torch, features, counts, train, fold, device)
            predictions[METHODS.index(method), ~train] = values
            diagnostics.append({"fold": fold, "method": method, **record})
        print(f"OneStopQA controls completed fold {fold + 1}/5", flush=True)

    losses = {method: row_losses(p, counts, correct) for method, p in zip(METHODS, predictions, strict=True)}
    totals = counts.sum(axis=1)
    all_totals = totals.sum()
    unique_families, groups = np.unique(families, return_inverse=True)
    family_counts = np.bincount(groups, weights=totals)
    draws = np.random.default_rng(43).integers(0, len(unique_families), size=(10000, len(unique_families)))
    contrasts = {}
    for metric, full in losses[METHODS[2]].items():
        difference = full - losses[METHODS[1]][metric]
        sums = np.bincount(groups, weights=difference)
        by_draw = sums[draws].sum(axis=1) / family_counts[draws].sum(axis=1)
        contrasts[metric] = {
            "difference": float(difference.sum() / all_totals),
            "95_ci": np.quantile(by_draw, [0.025, 0.975]).tolist(),
        }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", ids=np.array(ids), methods=np.array(METHODS), predictions=predictions)
    report = {
        "scope": "OneStop Prolific selected options, five held-out article-family folds.",
        "trials": int(all_totals),
        "families": len(unique_families),
        "human_correct": int(counts[np.arange(len(correct)), correct].sum()),
        "trained_device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "ridge_penalty": RIDGE,
        "timing_s": perf_counter() - started,
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("design", design_path), ("features", features_path))
        },
        "methods": {
            method: {metric: float(value.sum() / all_totals) for metric, value in values.items()}
            for method, values in losses.items()
        },
        "with_passage_minus_question_only": contrasts,
        "bootstrap": "10,000 paired held-out article-family draws; fixed fitted predictions and reader sample.",
        "diagnostics": diagnostics,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("design", "features", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    args = parser.parse_args()
    evaluate(args.design, args.features, args.output, args.device)
