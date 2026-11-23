"""Compress three measured human responses into a shared slope and profiled likelihood curvature."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit

from research.reading_validation.joint_data import DEFAULT_CACHE, auxiliary
from research.reading_validation.joint_fit import (
    RIDGE,
    fit,
    normalized_hidden,
    observations,
)
from research.reading_validation.local_forecast import digest
from research.reading_validation.noun_material import MODEL
from research.reading_validation.relay_summary_material import OUT


def coordinates(surface, hidden, source):
    sf = (surface - source["zero.basis.surface_mean"]) / source[
        "zero.basis.surface_scale"
    ]
    x = (normalized_hidden(hidden) - source["zero.basis.hidden_mean"]) @ source[
        "zero.basis.projection"
    ][:, -24:]
    return np.column_stack((sf, x))


def main():
    start = perf_counter()
    metadata = json.loads(
        (DEFAULT_CACHE / "reading-validation/readme/materials.json").read_bytes()
    )
    source, receipt = auxiliary(DEFAULT_CACHE, metadata)
    if len(source) != 3:
        raise ValueError("Expected three authorized human inputs")
    with np.load(MODEL, allow_pickle=False) as fit_file:
        inputs = {
            k: coordinates(s.surface, s.hidden, fit_file) for k, s in source.items()
        }
    model = fit(list(source.values()), inputs)
    width = model.slope.size
    curvature = RIDGE * np.eye(width)
    counts = {}
    for k, s in source.items():
        subset = s.role == 0
        y, w = observations(s, subset)
        x = inputs[k][subset]
        means = (x @ model.slope)[:, None] + model.offsets[k]
        ps = expit(means)
        prec = np.ones(y.shape) if s.continuous else ps * (1 - ps)
        weights = prec * w[:, None] / (len(source) * y.shape[1])
        for weight in weights.T:
            center = np.average(x, axis=0, weights=weight)
            residual = x - center
            curvature += (residual.T * weight) @ residual
        counts[k] = {
            "training_rows": int(subset.sum()),
            "response_coordinates": y.shape[1],
            "objective": "weighted Gaussian"
            if s.continuous
            else "count-weighted cumulative Bernoulli",
        }
    precision = curvature / (np.trace(curvature) / width)
    eigen = np.linalg.eigvalsh(precision)
    if eigen.min() <= 0 or abs(float(np.trace(precision)) / width - 1) > 1e-10:
        raise ArithmeticError("Invalid profiled prior curvature")
    p = OUT / "collective-prior.npz"
    np.savez_compressed(p, center=model.slope, precision=precision)
    manifest = {
        "protocol_sha256": digest(
            Path(__file__).with_name("relay-summary-joint-protocol.md")
        ),
        "source": receipt,
        "source_counts": counts,
        "source_prior_sha256": digest(p),
        "features": width,
        "isotropic_regularizer": RIDGE,
        "eigenvalue_range": [float(eigen.min()), float(eigen.max())],
        "condition_number": float(eigen.max() / eigen.min()),
        "source_convergence": model.convergence,
        "elapsed_seconds": perf_counter() - start,
        "code_sha256": digest(Path(__file__)),
        "cefr_labels_used": False,
        "heldout_target_read": False,
    }
    (OUT / "collective-source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(counts, model.convergence, eigen[[0, -1]])


if __name__ == "__main__":
    main()
