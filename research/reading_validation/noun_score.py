"""Score paired held-family forecasts, retaining cloze and perceived difficulty separately."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.reading_validation.local_forecast import digest
from research.reading_validation.noun_data import OUT, design
from research.reading_validation.noun_fit import ARMS, ENDPOINTS, FIXED

SEED = 726031
DRAWS = 10_000


class Families:
    def __init__(self, strata):
        self.strata = np.asarray(strata)
        rng = np.random.default_rng(SEED)
        draws = []
        for size in sorted(set(self.strata)):
            ids = np.flatnonzero(self.strata == size)
            draws.append(rng.choice(ids, size=(DRAWS, len(ids)), replace=True))
        if not draws:
            raise ValueError("No eligible family")
        self.draws = np.concatenate(draws, axis=1)

    def interval(self, x):
        x = np.asarray(x, dtype=float)
        valid = np.isfinite(x)
        counts = valid[self.draws].sum(axis=1)
        ok = counts > 0
        if not valid.any():
            return {"mean": None, "ci95": [None, None], "families": 0}
        if ok.mean() < 0.99:
            raise ValueError(
                "Insufficient informative families for the stratified bootstrap"
            )
        b = np.asarray(np.where(valid, x, 0)[self.draws].sum(axis=1)[ok] / counts[ok])
        return {
            "mean": float(np.mean(x[valid])),
            "ci95": np.quantile(b, [0.025, 0.975]).tolist(),
            "families": int(valid.sum()),
        }


def differences(a, intervention):
    # axes are family, split (no / yes), pseudoword (no / yes)
    return (
        a[:, 1, :] - a[:, 0, :] if intervention == "split" else a[:, :, 1] - a[:, :, 0]
    )


def valid_average(x, eligible):
    n = eligible.sum(axis=1)
    y = np.full(len(x), np.nan)
    np.divide(np.where(eligible, x, 0).sum(axis=1), n, out=y, where=n > 0)
    return y


def estimate_methods(y, signals, bootstrap, intervention):
    truth = differences(y, intervention)
    out = {}
    for k, signal in (("observed", y), *signals.items()):
        d = differences(signal, intervention)
        eligible = truth != 0
        win = (1 + np.sign(truth) * np.sign(d)) / 2
        vals = {
            "signed_change": bootstrap.interval(d.mean(axis=1)),
            "condition_0_change": bootstrap.interval(d[:, 0]),
            "condition_1_change": bootstrap.interval(d[:, 1]),
        }
        if k != "observed":
            vals.update(
                {
                    "direction": bootstrap.interval(valid_average(win, eligible)),
                    "condition_0_direction": bootstrap.interval(
                        np.where(eligible[:, 0], win[:, 0], np.nan)
                    ),
                    "condition_1_direction": bootstrap.interval(
                        np.where(eligible[:, 1], win[:, 1], np.nan)
                    ),
                }
            )
            if k in ARMS:
                vals["contrast_mse"] = bootstrap.interval(
                    np.mean((d - truth) ** 2, axis=1)
                )
        out[k] = vals
    out["human_non_tied_comparisons"] = int((truth != 0).sum())
    return out


def run(*, independently_varying=False):
    output = OUT / "varying-blocks" if independently_varying else OUT
    lock = json.loads((output / "forecast-seal.json").read_bytes())
    material = json.loads((OUT / "material-audit.json").read_bytes())
    for p, v in json.loads((OUT / "material-sensitivity-lock.json").read_bytes())[
        "files"
    ].items():
        if digest(Path(p)) != v:
            raise ValueError("Material subset protocol changed")
    if (
        digest(output / "predictions.npz") != lock["predictions_sha256"]
        or digest(Path(__file__)) != lock["score_script_sha256"]
    ):
        raise ValueError("Score program or forecasts changed")
    records = {r["id"]: r for r in design()}
    scores = {}
    with np.load(output / "predictions.npz", allow_pickle=False) as bank:
        ordered = [records[k] for k in bank["id"].tolist()]
        families = bank["family"].reshape(-1, 4)[:, 0]
        if not np.array_equal(
            np.asarray([r["family"] for r in ordered]), bank["family"]
        ):
            raise ValueError("Response-family alignment changed")
        gr = np.asarray([r["gram"] for r in ordered])[::4]
        clean = np.asarray(
            [
                f not in {r["family"] for r in material["textual_geometry_flags"]}
                for f in families
            ]
        )
        for mi, key in enumerate(ENDPOINTS):
            y = np.asarray([r[key] for r in ordered]).reshape(-1, 2, 2)
            methods = {k: bank[k][:, mi].reshape(-1, 2, 2) for k in ARMS}
            methods.update({k: bank[k].reshape(-1, 2, 2) for k in FIXED})
            panels = {
                "primary": np.ones(len(gr), bool),
                "clean_material": clean,
                **{f"length_{i}": gr == i for i in sorted(set(gr))},
            }
            scores[key] = {}
            for group, eligible in panels.items():
                yy = y[eligible]
                ss = {k: s[eligible] for k, s in methods.items()}
                b = Families(gr[eligible])
                errors = {
                    k: ((s - yy) ** 2).mean(axis=(1, 2))
                    for k, s in ss.items()
                    if k in ARMS
                }
                mse = {k: b.interval(e) for k, e in errors.items()}
                for control, extended in (
                    ("surface", "extended"),
                    ("surface_condition", "extended_condition"),
                    ("condition", "extended_condition"),
                    ("mean", "surface"),
                ):
                    mse[f"{control}_minus_{extended}"] = b.interval(
                        errors[control] - errors[extended]
                    )
                scores[key][group] = {
                    "families": int(eligible.sum()),
                    "mse": mse,
                    **{
                        k: estimate_methods(yy, ss, b, k)
                        for k in ("split", "pseudoword")
                    },
                }
    result = {
        "scores": scores,
        "draws": DRAWS,
        "seed": SEED,
        "primary_family_stratified_mse_gain": scores["difficulty"]["primary"]["mse"][
            "surface_minus_extended"
        ],
        "primary_passed": scores["difficulty"]["primary"]["mse"][
            "surface_minus_extended"
        ]["ci95"][0]
        > 0,
        "forecast_sha256": lock["predictions_sha256"],
        "raw_response_file_sha256": json.loads((OUT / "input-seal.json").read_bytes())[
            "workbook_sha256"
        ],
    }
    (output / "scores.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        {k: result[k] for k in ("primary_family_stratified_mse_gain", "primary_passed")}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independently-varying", action="store_true")
    run(independently_varying=parser.parse_args().independently_varying)
