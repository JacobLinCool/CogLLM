"""Fit and seal sentence-difficulty forecasts before joining external test labels."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from research.reading_validation.joint_data import (
    DEFAULT_CACHE,
    auxiliary,
    digest,
    external,
    read,
)
from research.reading_validation.joint_fit import (
    Basis,
    archive_model,
    fit,
    mixture_weight,
    observations,
)


ROOT = Path(__file__).resolve().parent


def archived_basis(archive, name, basis):
    for key in ("hidden_mean", "projection", "surface_mean", "surface_scale"):
        archive[f"{name}.{key}"] = getattr(basis, key)


def loss_by_row(study, pred, where):
    target, weight = observations(study, where)
    squared = ((pred[where] - target) ** 2).mean(axis=1)
    if not study.continuous:
        # Restore the variance of the human response around its empirical proportion.
        squared += (target * (1 - target)).mean(axis=1)
    return float(np.average(squared, weights=weight))


def run(cache, extraction, output):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(f"Choose a fresh forecast directory: {output}")
    metadata = read(cache / "reading-validation/readme/materials.json")
    studies, audit = auxiliary(cache, metadata)
    target = external(cache, extraction, metadata)
    if read(extraction / "extraction.json")["revision"] != audit["revision"]:
        raise ValueError("External and auxiliary features used different backbones")
    n_test = target.role == 2
    if not np.isnan(target.response[n_test]).all():
        raise ValueError("Test outcomes reached the fit")
    archive, forecast, converged, diagnostics = {}, {}, {}, {}
    zero_basis = Basis.train(list(studies.values()))
    archived_basis(archive, "zero.basis", zero_basis)
    for design in ("surface", "qwen"):
        hidden = design == "qwen"
        features = {key: zero_basis.transform(s, hidden) for key, s in studies.items()}
        model = fit(list(studies.values()), features)
        name = "zero." + design
        archive_model(archive, name, model)
        converged[name] = model.convergence
        forecast[name] = zero_basis.transform(target, hidden)[n_test] @ model.slope

    studies["readme"] = target
    basis = Basis.train(list(studies.values()))
    archived_basis(archive, "fit.basis", basis)
    for design in ("surface", "qwen"):
        features = {
            key: basis.transform(s, design == "qwen") for key, s in studies.items()
        }
        pooled = fit(list(studies.values()), features)
        archive_model(archive, f"pooled.{design}", pooled)
        converged[f"pooled.{design}"] = pooled.convergence
        for name, s in studies.items():
            independent = fit([s], features)
            archive_model(archive, f"independent.{design}.{name}", independent)
            converged[f"independent.{design}.{name}"] = independent.convergence
            pooled_pred = pooled.predict(s, features[name])
            alone_pred = independent.predict(s, features[name])
            beta = mixture_weight(s, alone_pred, pooled_pred)
            archive[f"mix.{design}.{name}"] = np.asarray(beta)
            adaptive = alone_pred + beta * (pooled_pred - alone_pred)
            values = {
                "independent": alone_pred,
                "pooled": pooled_pred,
                "adaptive": adaptive,
            }
            diagnostics[f"{design}.{name}"] = {
                "mixture_weight": beta,
                "development": {
                    k: loss_by_row(s, p, s.role == 1) for k, p in values.items()
                },
                "train_rows": int((s.role == 0).sum()),
                "validation_rows": int((s.role == 1).sum()),
                "train_groups": len(set(np.asarray(s.groups)[s.role == 0])),
                "validation_groups": len(set(np.asarray(s.groups)[s.role == 1])),
            }
            if name == "readme":
                for mode, values_ in values.items():
                    forecast[f"{mode}.{design}"] = values_[n_test]
    y, weight = observations(target, target.role == 0)
    forecast["constant"] = np.tile(
        np.average(y, axis=0, weights=weight), (n_test.sum(), 1)
    )
    output.mkdir(parents=True)
    np.savez_compressed(output / "models.npz", **archive)
    np.savez_compressed(output / "forecasts.npz", **forecast)
    test_rows = [
        dict(id=r["id"], row=r["row"], domain=r["domain"], subdomain=r["subdomain"])
        for r in metadata["rows"]
        if r["split"] == "test"
    ]
    (output / "heldout.json").write_text(
        json.dumps(test_rows, ensure_ascii=False, indent=2) + "\n"
    )
    code = {
        name: digest(ROOT / name)
        for name in (
            "joint-protocol.md",
            "external_input.py",
            "external_forecast.py",
            "joint_data.py",
            "joint_fit.py",
            "joint_forecast.py",
        )
    }
    protected = Path(
        "research/results/adaptive-joint-fixed-retention-l4/guarded-fixed-1729/curve.pt"
    )
    seal = dict(
        inputs=audit,
        extraction_sha256=digest(extraction / "extraction.json"),
        workbook_sha256=metadata["workbook_sha256"],
        material_manifest_sha256=digest(
            cache / "reading-validation/readme/materials.json"
        ),
        protocol_and_code=code,
        external_overlap_removals=metadata["exclusions"],
        n_heldout=len(test_rows),
        diagnostics=diagnostics,
        convergence=converged,
        existing_curve_sha256=digest(protected),
        wall_seconds=time.monotonic() - start,
        files={
            name: digest(output / name)
            for name in ("forecasts.npz", "heldout.json", "models.npz")
        },
    )
    (output / "seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    return seal


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--extraction", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    opts = p.parse_args()
    receipt = run(opts.cache, opts.extraction, opts.output)
    print(
        json.dumps(
            {"heldout": receipt["n_heldout"], "seconds": receipt["wall_seconds"]}
        )
    )
