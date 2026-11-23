"""Score sealed forecasts and quantify uncertainty across the sampled domains."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from research.reading_validation.external_input import digest_string, workbook_rows
from research.reading_validation.joint_data import DEFAULT_CACHE, digest, read


def score(forecasts, folder, output, draws=10000):
    seal = read(forecasts / "seal.json")
    for file, expected in seal["files"].items():
        if digest(forecasts / file) != expected:
            raise ValueError("The training seal does not match the candidate forecasts")
    if digest(folder / "test.xlsx") != seal["workbook_sha256"]["test"]:
        raise ValueError("The test workbook changed after prespecification")
    heldout = read(forecasts / "heldout.json")
    actual = {
        n: (digest_string(r[4]), float(r[5])) for n, r in workbook_rows(folder, "test")
    }
    if any(actual[row["row"]][0] != row["id"] for row in heldout):
        raise ValueError("External stimulus order differs from the seal")
    y = np.array([actual[row["row"]][1] for row in heldout])
    if (y != np.floor(y)).any() or not np.all((1 <= y) & (y <= 6)):
        raise ValueError("Expected six ordered CEFR levels")
    group = np.array([row["domain"] for row in heldout])
    categories = np.arange(1, 6)
    binary = y[:, None] > categories
    methods = np.load(forecasts / "forecasts.npz", allow_pickle=False)
    values, means, ordinal_metrics = {}, {}, {}
    for k in methods.files:
        pred = methods[k]
        if not np.isfinite(pred).all() or len(pred) != len(y):
            raise ValueError(f"Invalid prediction for {k}")
        if pred.ndim == 1:
            means[k] = pred
            continue
        if (
            pred.shape != binary.shape
            or (np.diff(pred, axis=1) > 1e-9).any()
            or not np.all((0 <= pred) & (pred <= 1))
        ):
            raise ValueError(f"Forecast is not an ordinal distribution for {k}")
        means[k] = 1 + pred.sum(axis=1)
        values[k] = ((binary - pred) ** 2).mean(axis=1)
        probs = -np.diff(
            np.column_stack([np.ones(len(pred)), pred, np.zeros(len(pred))]), axis=1
        )
        ordinal_metrics[k] = dict(
            rps=float(values[k].mean()),
            mae_levels=float(np.abs(means[k] - y).mean()),
            mean_predicted_level=float(means[k].mean()),
            modal_accuracy=float(np.mean(1 + probs.argmax(axis=1) == y)),
        )
    groups = np.unique(group)
    members = [np.flatnonzero(group == g) for g in groups]
    rng = np.random.default_rng(20261122)
    samples = rng.integers(0, len(groups), size=(draws, len(groups)))

    # Average ranks of tied predictions and outcomes are recalculated in each resample.
    def rho(a, b):
        l, r = rankdata(a), rankdata(b)
        l -= l.mean()
        r -= r.mean()
        denom = np.sqrt(np.dot(l, l) * np.dot(r, r))
        return float(np.dot(l, r) / denom) if denom > 0 else None

    def interval(observed, samples_):
        estimates = np.asarray(samples_, dtype=np.float64)
        if not np.isfinite(estimates).all():
            raise ValueError("A bootstrap contrast was undefined")
        return dict(
            estimate=float(observed),
            interval95=np.quantile(estimates, [0.025, 0.975]).tolist(),
        )

    comparisons = {}
    for k in ("independent.surface", "independent.qwen", "pooled.qwen", "constant"):
        difference = values["adaptive.qwen"] - values[k]
        total = np.array([difference[m].sum() for m in members])
        sizes = np.array([len(m) for m in members])
        draws_ = total[samples].sum(axis=1) / sizes[samples].sum(axis=1)
        comparisons[f"adaptive.qwen_minus_{k}"] = interval(difference.mean(), draws_)
    corr = {k: rho(v, y) for k, v in means.items()}
    zero_delta = []
    zero_qwen, zero_surface = [], []
    for sample in samples:
        idx = np.concatenate([members[i] for i in sample])
        first, control = (
            rho(means[k][idx], y[idx]) for k in ("zero.qwen", "zero.surface")
        )
        zero_qwen.append(first)
        zero_surface.append(control)
        zero_delta.append(first - control)
    zero = {
        "qwen": interval(corr["zero.qwen"], zero_qwen),
        "surface": interval(corr["zero.surface"], zero_surface),
        "qwen_minus_surface": interval(
            corr["zero.qwen"] - corr["zero.surface"], zero_delta
        ),
    }
    by_domain = {}
    for lab, m in zip(groups, members, strict=True):
        by_domain[str(lab)] = dict(
            n=len(m),
            adaptive_rps=float(values["adaptive.qwen"][m].mean()),
            independent_rps=float(values["independent.qwen"][m].mean()),
            surface_rps=float(values["independent.surface"][m].mean()),
            human_mean=float(y[m].mean()),
            adaptive_mean=float(means["adaptive.qwen"][m].mean()),
        )
    result = dict(
        n_sentences=len(y),
        n_domains=len(groups),
        bootstrap_domain_draws=draws,
        forecast_seal_sha256=digest(forecasts / "seal.json"),
        mean_human_level=float(y.mean()),
        no_target_study_training_spearman=zero,
        test_spearman=corr,
        test_ordinal=ordinal_metrics,
        paired_test_rps_differences=comparisons,
        test_domains=by_domain,
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forecasts", type=Path, required=True)
    p.add_argument(
        "--folder", type=Path, default=DEFAULT_CACHE / "reading-validation/readme"
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    scored = score(args.forecasts, args.folder, args.output)
    print(
        json.dumps(
            {
                k: scored[k]
                for k in (
                    "n_sentences",
                    "no_target_study_training_spearman",
                    "paired_test_rps_differences",
                )
            }
        )
    )
