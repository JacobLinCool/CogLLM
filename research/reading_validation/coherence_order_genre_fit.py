"""Freeze independent genre and genre-plus-form controls on the primary development design."""

import json
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_data import OUT, read
from research.reading_validation.coherence_order_estimator import (
    equal_group_weights,
    fit,
    predict,
    prepare,
)
from research.reading_validation.coherence_order_fit import source_targets
from research.reading_validation.local_forecast import digest
from research.reading_validation.memory_data import h


def readouts():
    f = OUT / "form-features.npz"
    if read(f.with_suffix(".json"))["features"] != digest(f):
        raise ValueError("Form signatures differ")
    with np.load(f, allow_pickle=False) as z:
        idx = z["ids"]
        x = z["x"]
    mask = (idx[:, 3] == "italian") & (idx[:, 4] == "fit")
    xtr = x[mask]
    it = idx[mask]
    held = (idx[:, 3] == "italian") & (idx[:, 4] == "held")
    itest = idx[held]
    xtest = x[held]
    gt = source_targets()["italian"]
    levels = sorted(
        set(it[:, 2]), key=lambda g: h("coherence-genre-held-source-v1." + g)
    )
    fold = {g: i % 4 for i, g in enumerate(levels)}
    rows = np.array([fold[k] for k in it[:, 2]])
    oof = {}
    final = {}
    checks = []
    for turn in range(5):
        select = (rows != turn) if turn < 4 else np.ones(len(rows), bool)
        test = (it[rows == turn], xtr[rows == turn]) if turn < 4 else (itest, xtest)
        w = equal_group_weights(it[select, 2])
        center = w @ xtr[select]
        sd = np.sqrt(w @ ((xtr[select] - center) ** 2))
        sd = np.where(sd > 1e-6, sd, 1.0)
        a = prepare(it[select], (xtr[select] - center) / sd, "italian", gt)
        b = prepare(test[0], (test[1] - center) / sd, "italian")
        theta, note = fit([a])
        checks.append(note)
        dest = oof if turn < 4 else final
        dest.update(zip(b["keys"], predict(theta, b), strict=True))
    if len(oof) != len(set(it[:, 1])) or len(final) != len(set(itest[:, 1])):
        raise ValueError("Overlap in source folds")
    source = OUT / "form-preflight/predictions.npz"
    with np.load(source, allow_pickle=False) as z:
        q = [final[k] for k in z["italian.keys"]]
        err = np.max(abs(z["italian.form"] - q))
        if err > 1e-7:
            raise ValueError(
                "Independent Italian form control differs from existing forecasts"
            )
    return oof, final, checks, float(err)


def run():
    source = OUT / "genre-forecasts.json"
    feature = read(source)
    note = read(OUT / "genre-seal.json")
    if digest(source) != note["source"]:
        raise ValueError("Changing a genre forecast after seeing a grade is disallowed")
    oof, form, checks, err = readouts()
    g = read(OUT / "geometry.json")
    groups = {
        r["source_id"]: r["group"] for r in g.values() if r["source"] == "italian"
    }
    gt = source_targets()["italian"]

    def design(keys, pred, full):
        p = np.array([feature[k]["wiki_probability"] for k in keys])
        cols = [np.ones(len(keys)), p]
        if full:
            cols.append([pred[k] for k in keys])
        return np.column_stack(cols)

    train = sorted(oof)
    test = sorted(form)
    predictions = {"keys": np.array(test)}
    truth = np.array([gt[k] for k in train])
    w = equal_group_weights([groups[k] for k in train])
    fits = {}
    for arm, full in (("genre", False), ("genre_plus_form", True)):
        m = design(train, oof, full)
        t = design(test, form, full)
        beta = np.linalg.lstsq(m * np.sqrt(w[:, None]), truth * np.sqrt(w), rcond=None)[
            0
        ]
        predictions[arm] = np.clip(t @ beta, 1.0, 5.0)
        fits[arm] = beta.tolist()
    predictions["predicted_genre"] = np.array(
        ["wiki" if feature[k]["wiki_probability"] >= 0.5 else "ted" for k in test]
    )
    predictions["observed_genre"] = np.array(
        [feature[k]["matched_genre"] or "unidentified" for k in test]
    )
    p = OUT / "genre-controls.npz"
    np.savez_compressed(p, **predictions)
    rec = {
        "forecast": digest(p),
        "source_held_outcome_free": True,
        "coefficients": fits,
        "fold_fits": checks,
        "form_reproduction_mae_max": err,
        "training_passages": len(train),
        "held_passages": len(test),
        "registered_genre_question": digest(
            Path(__file__).with_name("coherence_order_genre_protocol.md")
        ),
    }
    (OUT / "genre-controls-seal.json").write_text(json.dumps(rec, indent=2) + "\n")
    print({k: v for k, v in rec.items() if k != "fold_fits"})


if __name__ == "__main__":
    run()
