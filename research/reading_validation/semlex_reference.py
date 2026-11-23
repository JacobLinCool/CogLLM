"""Check the transfer estimator's coordinates against previously frozen source forecasts."""

import json

import numpy as np
from scipy.special import logit

from research.reading_validation.local_forecast import digest
from research.reading_validation.nns_fit import all_features
from research.reading_validation.nns_inference import WordReport
from research.reading_validation.nns_rewrite_features import comparisons
from research.reading_validation.semlex_forecast import (
    NATIVE,
    TRAIN_CACHE,
    TRAIN,
    source_predictions,
)
from research.reading_validation.semlex_material import OUT


def run():
    w = WordReport.load()
    with np.load(TRAIN_CACHE / "word-features.npz") as f:
        bank = dict(f)
    pairs = comparisons(bank, "held", outcomes=False)
    with np.load(TRAIN / "forecasts.npz") as f:
        raw = dict(f)
    with np.load(TRAIN / "weights.npz") as f:
        weights = dict(f)
    if not np.array_equal(raw["r_keys"], pairs["keys"]):
        raise ValueError("Source pair mapping changed")
    n = len(bank["form"])
    source = np.asarray([w.word_rates.get(stem, (0, 0)) for stem in bank["stem"]])
    y, t = source.T
    rate = np.column_stack(
        [
            logit(np.clip((y + 20 * w.prevalence) / (t + 20), 1e-7, 1 - 1e-7)),
            np.log1p(t),
        ]
    )
    at = np.column_stack([np.arange(n), np.zeros(n, int)])
    x = [
        all_features(bank, at, rate, kind, False) for kind in ("lexical", "contextual")
    ]
    seal = json.loads((OUT / "source-calibration.json").read_bytes())
    a = seal["convex_weight_on_shared_contextual"]
    got = source_predictions(x, bank["surprisal"], pairs["pair"], weights, a)
    base = raw["reference.learned"]
    joint = raw["r.contextual.adaptive"]
    both = (1 - a) * base + a * joint
    want = np.column_stack(
        [q[:, 0] - q[:, 1] for q in (both, base, joint, raw["r.lexical.adaptive"])]
        + [raw["reference.target"], raw["zero.rarity"][:, 1] - raw["zero.rarity"][:, 0]]
    )
    errors = np.max(abs(got - want), axis=0)
    if max(errors) > 1e-7:
        raise ValueError(
            f"External transfer implementation diverged from source: {errors}"
        )
    rec = {
        "source_calibration_sha256": digest(OUT / "source-calibration.json"),
        "source_pairs": len(got),
        "source_forecasts_compared": int(got.size),
        "max_abs_difference": dict(zip(NATIVE, errors.tolist(), strict=True)),
    }
    (OUT / "transfer-parity.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(rec)


if __name__ == "__main__":
    run()
