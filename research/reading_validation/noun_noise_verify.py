"""Independently construct integer-weight witnesses for the four cloze-rate extrema."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from research.reading_validation.local_forecast import digest
from research.reading_validation.noun_data import OUT, design
from research.reading_validation.noun_noise import MAX_CELL, TOL, TRIALS


def main():
    original = OUT / "human-measurement.json"
    ref = json.loads(original.read_bytes())
    data = design()
    nitems = len(data)
    a = []
    for i, r in enumerate(data):
        for n in range(1, MAX_CELL + 1):
            er, df = n * r["error"], n * r["difficulty"]
            if abs(er - round(er)) <= TOL and abs(df - round(df)) <= TOL:
                a.append((i, n, round(er), r["split"]))
    ix = np.arange(len(a))
    item, nobs, nerr, was_split = np.array(a, int).T
    integrality = np.ones(len(a))
    bounds = Bounds(0, 1)
    certificates = []
    for key, split in (("split", True), ("unsplit", False)):
        chosen = was_split == split
        for side, v in zip(
            ("minimum", "maximum"), ref[key]["weighted_error_bounds"], strict=True
        ):
            fraction = Fraction(v).limit_denominator(TRIALS)
            last = np.where(
                chosen, nerr * fraction.denominator - nobs * fraction.numerator, 0
            )
            mat = coo_matrix(
                (
                    np.r_[np.ones(len(ix)), nobs, last],
                    (
                        np.r_[
                            item, np.full(len(ix), nitems), np.full(len(ix), nitems + 1)
                        ],
                        np.tile(ix, 3),
                    ),
                ),
                shape=(nitems + 2, len(ix)),
            ).tocsc()
            target = np.r_[np.ones(nitems), TRIALS, 0]
            ans = milp(
                np.zeros(len(ix)),
                integrality=integrality,
                bounds=bounds,
                constraints=LinearConstraint(mat, target, target),
                options={"time_limit": 90},
            )
            if not ans.success:
                raise ValueError(f"No integer witness for {key} {side}: {ans.message}")
            z = np.rint(ans.x).astype(int)
            if not np.array_equal(mat @ z, target) or np.any((z != 0) & (z != 1)):
                raise ArithmeticError(
                    "Reported solution failed exact integer constraints"
                )
            trial_counts = nobs[z == 1]
            ce = int((chosen * nerr) @ z)
            ct = int((chosen * nobs) @ z)
            if sum(trial_counts) != TRIALS or Fraction(ce, ct) != fraction:
                raise ArithmeticError("Constructive witness failed ratio")
            certificates.append(
                {
                    "condition": key,
                    "bound": side,
                    "errors": ce,
                    "assessments": ct,
                    "rate": float(fraction),
                    "global_assessments": int(sum(trial_counts)),
                    "counts_in_workbook_order": trial_counts.tolist(),
                }
            )
    result = {
        "source_bounds_sha256": digest(original),
        "verification_code_sha256": digest(Path(__file__)),
        "method": "Binary choice of one denominator per row; integer constraints enforce total, both released cell means, and the signed numerator difference for the exact target rational.",
        "witnesses": certificates,
    }
    (OUT / "trial-count-witnesses.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        [
            (r["condition"], r["bound"], r["errors"], r["assessments"])
            for r in certificates
        ]
    )


if __name__ == "__main__":
    main()
