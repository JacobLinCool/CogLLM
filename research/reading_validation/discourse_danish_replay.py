"""Replay Danish within-domain human ordering directly from the original public table and sealed predictions."""

import argparse
import csv
import hashlib
import json
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ARCHIVE_SHA256 = "f8c8d7bab8a9a065f6fb28b3d9abecc28a0b2aca1dd6321ab4bf2c9e5fcdff90"
ARMS = (
    "form",
    "surprisal",
    "representation",
    "shared",
    "adaptive",
    "danish_form",
    "danish_lexical",
    "domain_only",
)


def literal_order(y, p, w):
    """Use the full human-ordered pair matrix, with multiplicity at both ends for each resample."""
    actual = y[:, None] > y[None, :]
    predicted = p[:, None] - p[None, :]
    right = actual * ((predicted > 0) + 0.5 * (predicted == 0))
    comparable = actual.astype(float)
    result = []
    for i in range(0, len(w), 300):
        weights = w[i : i + 300]
        numerator = (weights * (weights @ right)).sum(1)
        denom = (weights * (weights @ comparable)).sum(1)
        result.append(
            np.divide(
                numerator, denom, out=np.full(len(weights), np.nan), where=denom > 0
            )
        )
    return np.concatenate(result)


def run(archive, package):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Unexpected original DDisCo archive")
    seal = json.loads((package / "prediction-seal.json").read_bytes())
    if (
        hashlib.sha256((package / "forecasts.npz").read_bytes()).hexdigest()
        != seal["predictions"]
    ):
        raise ValueError("Unrecognized prospective Danish predictions")
    with (
        ZipFile(archive) as f,
        f.open("ddisco.test.tsv") as g,
        TextIOWrapper(g, encoding="utf8", newline="") as s,
    ):
        rows = {
            hashlib.sha256(d["text"].encode()).hexdigest(): (
                int(d["rating"]),
                d["domain"],
            )
            for d in csv.DictReader(s, delimiter="\t")
        }
    answers = json.loads((package / "scores.json").read_bytes())
    with np.load(package / "forecasts.npz", allow_pickle=False) as forecast:
        keys = forecast["keys"]
        domains = forecast["domain"]
        groups = forecast["group"]
        target = np.asarray([rows[k][0] for k in keys])
        expected_domains = [rows[k][1] for k in keys]
        if not np.array_equal(expected_domains, domains):
            raise ValueError(
                "Domain labels in forecast and original Danish source differ"
            )
        domain_names = np.unique(domains)
        by_domain = {}
        for g, seed in zip(
            domain_names,
            np.random.SeedSequence(378283).spawn(len(domain_names)),
            strict=True,
        ):
            keep = domains == g
            distinct, inverse = np.unique(groups[keep], return_inverse=True)
            n = len(distinct)
            draw = np.random.default_rng(seed).multinomial(
                n, np.full(n, 1 / n), size=10000
            )
            multipliers = np.vstack([np.ones(keep.sum()), draw[:, inverse]])
            by_domain[g] = {
                k: literal_order(target[keep], forecast[k][keep], multipliers)
                for k in ARMS
            }
    overall = {k: np.mean([v[k] for v in by_domain.values()], axis=0) for k in ARMS}
    errors = []
    point_estimates = {}
    for domain, estimates in {**by_domain, "domain_equal": overall}.items():
        actual = (
            answers["domain_equal"]
            if domain == "domain_equal"
            else answers["domains"][domain]
        )
        for arm in ARMS:
            for panel, pred in [("by_model", estimates[arm])] + (
                [("adaptive_gain_over", estimates["adaptive"] - estimates[arm])]
                if arm != "adaptive"
                else []
            ):
                reference = actual[panel][arm]["concordance"]
                numerical = np.r_[pred[0], np.nanquantile(pred[1:], [0.025, 0.975])]
                errors.extend(
                    abs(
                        numerical
                        - np.array([reference["estimate"], *reference["ci95"]])
                    )
                )
                if domain == "domain_equal" and panel == "by_model":
                    point_estimates[arm] = float(pred[0])
    if not np.all(np.isfinite(errors)) or max(errors) > 1e-10:
        raise ArithmeticError(
            "Independent ordered-pair reconstruction differs from the released report"
        )
    print(
        json.dumps(
            {
                "estimated_concordances_and_paired_intervals": len(errors) // 3,
                "maximum_absolute_difference": max(errors),
                "source_passages": len(rows),
                "scored_passages": len(keys),
                "macro_within_domain_concordance": point_estimates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-zip", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    args = p.parse_args()
    run(args.original_zip, args.package)
