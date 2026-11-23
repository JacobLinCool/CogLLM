"""Export serving metadata and replay pooled observation outputs against original held measurements."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from research.adaptive_joint.cache import DEFAULT_RAW
from research.backbone_transfer.extract import digest
from research.reading_validation.borrowing_data import EVALUATE, PACKET
from research.reading_validation.borrowing_global_families import load, read_experts
from research.reading_validation.borrowing_inference import (
    DEFAULT,
    DEFAULT_PROJECTION,
    SCHEMA,
    AdaptiveObservationReader,
    CategoryForecast,
)
from research.reading_validation.borrowing_prior import SEEDS, read, write
from research.reading_validation.borrowing_score import GeometricPool


def metadata(tasks):
    src = read(PACKET.parent / "receipt.json")
    geometry = read(DEFAULT / "geometry.json")
    idx = {v["name"]: v for v in read(PACKET / "index.json")["tasks"]}
    schema = {}
    for t in tasks:
        s = idx[t.name]
        schema[t.name] = {
            "source": t.source,
            "likelihood": t.likelihood,
            "states": list(t.states),
            "max_positions": t.coordinates.shape[1],
            "coordinates": s["coordinates"],
            "output_width": t.output_width,
            "labels": s["labels"],
            "center": t.center,
            "spread": t.spread,
            "held_rows": geometry["registered_by_endpoint"][t.name]["registered_roles"][
                EVALUATE
            ],
        }
    weights = {}
    for seed in SEEDS:
        fs = read(DEFAULT / f"seed-{seed}" / "fit.json")["files"]
        weights[str(seed)] = {
            k: fs[k + "-parameters.npz"] for k in ("shared", "private")
        }
    write(
        SCHEMA,
        {
            "schema_version": 1,
            "qwen_revision": src["revision"],
            "projection_checkpoint_sha256": src["seed_sha256"],
            "previous_projection_training": "Previously supervised portable token readout. The fixed projection overlaps two subsequently re-held Frick training families. See the linked input receipt and source-level sensitivity.",
            "seeds": list(SEEDS),
            "calibration_sha256": digest(DEFAULT / "forecast-freeze.json"),
            "weights": weights,
            "endpoints": schema,
        },
    )


def actual_nll(t, row, forecast):
    y = t.response[row].double().numpy()
    if isinstance(forecast, CategoryForecast):
        # Work in probability space only at positive-mass labels to avoid log(0) at an unobserved risk interval.
        p = forecast.probabilities
        if np.max(np.abs(p.sum(axis=1) - 1)) > 1e-12 or (p[y > 0] <= 0).any():
            raise ArithmeticError("Invalid held forecast probability mass")
        return -(y * np.log(np.where(y > 0, p, 1.0))).sum(axis=1)
    mu = (forecast.mean - t.center) / t.spread
    var = forecast.covariance / (t.spread**2)
    if np.linalg.eigvalsh(var).min() <= 0:
        raise ArithmeticError("Nonpositive held covariance")
    if t.likelihood == "paired_gaussian":
        delta = y - mu
        return np.log(2 * np.pi) + 0.5 * (
            np.linalg.slogdet(var)[1]
            + np.einsum("bi,bij,bj->b", delta, np.linalg.inv(var), delta)
        )
    if t.likelihood == "gaussian":
        count, target, within = y.T
    else:
        count, target, within = 1, y[:, 0], 0.0
    return (
        count
        * 0.5
        * (
            np.log(2 * np.pi * var[:, 0, 0])
            + ((mu[:, 0] - target) ** 2 + within) / var[:, 0, 0]
        )
    )


def main():
    torch.set_num_threads(1)
    _, tasks, parts, _ = load()
    metadata(tasks)
    compact = np.load(PACKET / "states.npy", mmap_mode="r", allow_pickle=False)
    root = DEFAULT
    results = {}
    first = DEFAULT_RAW / "shards" / "shard-0000" / "states.npy"
    raw = np.load(first, mmap_mode="r", allow_pickle=False)
    sel = np.linspace(0, len(raw) - 1, 71).astype(int)
    coverage = set()
    for seed in SEEDS:
        reader = AdaptiveObservationReader(seed)
        projected = reader.project_hidden(
            torch.from_numpy(raw[sel].copy()), revision=reader.revision
        ).numpy()
        projection_abs = float(np.max(np.abs(projected - compact[sel])))
        if projection_abs > 1e-5:
            raise ArithmeticError(
                "Previously computed Qwen hidden states do not recover the serving projection"
            )
        stats = {}
        examples = {}
        total = 0
        with ExitStack() as stack:
            banks = read_experts(stack, seed, "development")
            for t in tasks:
                idx = np.flatnonzero(parts[t.name] == EVALUATE)
                if not len(idx):
                    continue
                pool = GeometricPool(t, idx, *(v[t.name] for v in banks))
                expected = pool.loss(reader.coefficients[t.name])
                error = 0.0
                weighted = 0.0
                ntrials = 0.0
                for pos in range(0, len(idx), 512):
                    real = idx[pos : pos + 512]
                    locations = t.coordinates[real].numpy()
                    h = np.array(compact[np.maximum(0, locations)], copy=True)
                    se = (
                        t.response[real, 1].double().numpy() * t.spread
                        if t.likelihood == "known_error_gaussian"
                        else None
                    )
                    pred = reader.predict_projected(
                        t.name,
                        torch.from_numpy(h),
                        available=torch.from_numpy(locations >= 0),
                        standard_errors=se,
                    )
                    loss = actual_nll(t, real, pred)
                    diff = loss - expected[pos : pos + len(real)]
                    error = max(error, float(np.max(np.abs(diff) / t.trials[real])))
                    weighted += float(np.sum(diff))
                    ntrials += float(np.sum(t.trials[real]))
                    if t.name not in examples:
                        if isinstance(pred, CategoryForecast):
                            examples[t.name] = {
                                "forecast_type": "finite_categories",
                                "labels": list(pred.labels),
                                "probabilities": pred.probabilities[0].tolist(),
                            }
                        else:
                            examples[t.name] = {
                                "forecast_type": "native_scale_gaussian",
                                "mean": pred.mean[0].tolist(),
                                "covariance": pred.covariance[0].tolist(),
                            }
                if error > 0.00004:
                    raise ArithmeticError(
                        f"Typed output differs from held density at {t.name}: {error}"
                    )
                stats[t.name] = {
                    "rows": len(idx),
                    "likelihood": t.likelihood,
                    "max_absolute_per_unit_nll_error": error,
                    "mean_signed_per_unit_nll_error": weighted / ntrials,
                }
                coverage.add(t.likelihood)
                total += len(idx)
        if len(coverage) != 10 or total != 13484:
            raise ValueError("Missing study or observation channel")
        results[str(seed)] = {
            "held_rows": total,
            "evaluated_endpoints": len(stats),
            "max_density_loss_error": max(
                x["max_absolute_per_unit_nll_error"] for x in stats.values()
            ),
            "real_hidden_projection_max_error": projection_abs,
            "endpoints": stats,
            "example_native_forecasts": examples,
        }
    dest = root / "serving-parity.json"
    write(
        dest,
        {
            "registered_endpoint_schemas": len(tasks),
            "observed_likelihoods": sorted(coverage),
            "per_seed": results,
            "projection_rows_replayed_per_seed": len(sel),
            "raw_Qwen_shard_sha256": digest(first),
            "projection_sha256": digest(DEFAULT_PROJECTION),
            "inference_sha256": digest(
                Path(__file__).with_name("borrowing_inference.py")
            ),
            "serving_schema_sha256": digest(SCHEMA),
        },
    )
    print(
        "Verified direct observation forecasts",
        [
            (s, z["max_density_loss_error"], z["real_hidden_projection_max_error"])
            for s, z in results.items()
        ],
    )


if __name__ == "__main__":
    main()
