"""Independent CPU computation for the noun contrasts using only assigned conditions or surface cues."""

import json
from pathlib import Path

import numpy as np

from research.reading_validation.joint_data import surface
from research.reading_validation.local_forecast import digest
from research.reading_validation.noun_data import OUT, design
from research.reading_validation.noun_fit import ENDPOINTS, RIDGE
from research.reading_validation.noun_score import Families, differences


def main():
    materials = json.loads((OUT / "material-audit.json").read_bytes())
    rows = sorted(
        (r for r in design() if r["family"] in materials["folds"]),
        key=lambda r: (r["family"], r["split"], r["pseudo"]),
    )
    y = np.array([[r[k] for k in ENDPOINTS] for r in rows])
    sf = np.array([surface(r["text"], "en") for r in rows])
    c = np.eye(12)[
        [4 * (r["gram"] - 2) + 2 * int(r["split"]) + int(r["pseudo"]) for r in rows]
    ]
    fold = np.asarray([materials["folds"][r["family"]] for r in rows])
    strata = [r["gram"] for r in rows][::4]
    b = Families(strata)
    results = {}
    predictions = {}
    for arm, xx in (
        ("condition", c),
        ("surface", sf),
        ("surface_condition", np.c_[sf, c]),
    ):
        pred = np.empty_like(y)
        for f in range(5):
            test = fold == f
            x = xx[~test]
            target = y[~test]
            xmean = x.mean(0)
            xsd = np.where(x.std(0) > 1e-9, x.std(0), 1)
            center = target.mean(0)
            z = (x - xmean) / xsd
            local = z - np.repeat(z.reshape(-1, 4, z.shape[1]).mean(1), 4, axis=0)
            local_y = target - np.repeat(target.reshape(-1, 4, 2).mean(1), 4, axis=0)
            rhs = z.T @ (target - center) + local.T @ local_y
            sigma = z.T @ z + local.T @ local + 2 * len(x) * RIDGE * np.eye(x.shape[1])
            pred[test] = (xx[test] - xmean) / xsd @ np.linalg.solve(sigma, rhs) + center
        predictions[arm] = pred
        cols = {}
        for j, endpoint in enumerate(ENDPOINTS):
            yhat = pred[:, j].reshape(-1, 2, 2)
            ygold = y[:, j].reshape(-1, 2, 2)
            cols[endpoint] = {
                "mse": b.interval(((yhat - ygold) ** 2).mean(axis=(1, 2))),
                "split_change": b.interval(differences(yhat, "split").mean(1)),
                "pseudoword_change": b.interval(
                    differences(yhat, "pseudoword").mean(1)
                ),
            }
        results[arm] = cols
    output = {
        "source_code_sha256": digest(Path(__file__)),
        "fixed_estimator_sha256": digest(OUT / "estimator-lock.json"),
        "numerical_method": "independently formed within-family and level covariance normal equations",
        "results": results,
    }
    if (OUT / "predictions.npz").exists():
        with np.load(OUT / "predictions.npz", allow_pickle=False) as z:
            if z["id"].tolist() != [r["id"] for r in rows]:
                raise ValueError("Different held-family row order")
            discrepancy = {
                arm: float(np.max(abs(val - z[arm])))
                for arm, val in predictions.items()
            }
            if max(discrepancy.values()) > 1e-10:
                raise ArithmeticError(
                    "Cross-fit disagrees with independent surface computation"
                )
            output["max_abs_difference_from_joint_pipeline"] = discrepancy
    (OUT / "surface-replay.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        {
            k: {e: (r["mse"]["mean"], r["split_change"]["mean"]) for e, r in v.items()}
            for k, v in results.items()
        }
    )


if __name__ == "__main__":
    main()
