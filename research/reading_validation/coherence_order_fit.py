"""Fit source-derived passage and boundary probability models and seal held predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_data import (
    HIST,
    ITALIAN,
    OUT,
    PLAN,
    csv_rows,
    read,
)
from research.reading_validation.coherence_order_estimator import (
    equal_group_weights,
    fit,
    mixture,
    normalize,
    predict,
    prepare,
)
from research.reading_validation.coherence_order_features import (
    BASE_COLUMNS,
    PRED_COLUMNS,
)
from research.reading_validation.local_forecast import digest
from research.reading_validation.transition_external_outcomes import votes


def source_targets(held=False):
    imap = {
        r["ID"]: float(r["SCORE"])
        for r in csv_rows(ITALIAN / ("held.tsv" if held else "train.tsv"))
    }
    if not all(1 <= v <= 5 for v in imap.values()):
        raise ValueError("Unknown Italian panel scale")
    smap = {}
    for r in read(HIST / "stimuli.json")["material"]:
        if r["source"] in ("cohesentia", "decor") and (
            (r["split"] in ("holdout", "external")) == held
        ):
            s, n, _, _ = votes(r)
            smap[r["id"]] = (s, n)
    return {"english": smap, "italian": imap}


def run(form_only):
    target = source_targets()
    feature_path = OUT / ("form-features.npz" if form_only else "native-features.npz")
    if digest(feature_path) != read(feature_path.with_suffix(".json"))["features"]:
        raise ValueError("Features differ from extraction")
    with np.load(feature_path, allow_pickle=False) as f:
        ids, x = f["ids"], f["x"]
    x, params = normalize(ids, x)
    audit = {}
    all_pred = {lang: {} for lang in ("english", "italian")}
    keys = {}
    arms = {"form": len(BASE_COLUMNS)}
    if not form_only:
        arms.update(
            surprisal=len(BASE_COLUMNS) + len(PRED_COLUMNS), representation=x.shape[1]
        )
    for name, size in arms.items():
        z = x[:, :size]
        by = {}
        for lang in ("english", "italian"):
            select = (
                (ids[:, 3] == "italian")
                if lang == "italian"
                else (ids[:, 3] != "italian")
            )
            for role in ("fit", "dev", "held"):
                row = select & (ids[:, 4] == role)
                q = prepare(
                    ids[row], z[row], lang, None if role == "held" else target[lang]
                )
                by[lang, role] = q
                if role == "held":
                    keys[lang] = q["keys"]
                    all_pred[lang]["group"] = q["group"]
                    all_pred[lang]["document"] = q["doc"]
            p, notes = fit([by[lang, "fit"]])
            audit[f"{name}.{lang}"] = notes
            params[f"{name}.{lang}"] = p
            all_pred[lang][name] = predict(p, by[lang, "held"])
        if name == ("form" if form_only else "representation"):
            joint, notes = fit([by["english", "fit"], by["italian", "fit"]])
            audit[name + ".shared"] = notes
            params[name + ".shared"] = joint
            for lang in ("english", "italian"):
                dev = by[lang, "dev"]
                test = by[lang, "held"]
                base = predict(params[name + "." + lang], dev)
                fraction = mixture(dev, base, predict(joint, dev))
                a = predict(params[name + "." + lang], test)
                b = predict(joint, test)
                all_pred[lang]["shared"] = b
                all_pred[lang]["adaptive"] = (1 - fraction) * a + fraction * b
                audit[lang + ".mixture"] = fraction
                params[lang + ".mixture"] = fraction
    note = read(OUT / "official-italian-seal.json")
    official = OUT / "official-italian.json"
    if digest(official) != note["forecast"]:
        raise ValueError("Unexpected official Jaccard reference")
    jaccard = {r["id"]: r["official"] for r in read(official)}
    g = read(OUT / "geometry.json")
    train = [
        r
        for r in g.values()
        if r["source"] == "italian" and r["role"] == "fit" and r["source_id"] in jaccard
    ]
    a = np.array([jaccard[r["source_id"]] for r in train])
    b = np.array([target["italian"][r["source_id"]] for r in train])
    w = equal_group_weights([r["group"] for r in train])
    u = a - w @ a
    slope = max(0.0, (w * u) @ b / ((w * u) @ u))
    base = w @ b - slope * (w @ a)
    jk = np.array([jaccard.get(s, np.nan) for s in keys["italian"]])
    all_pred["italian"].update(
        official_rank=jk, official_scaled=np.clip(base + slope * jk, 1, 5)
    )
    fitit = by["italian", "fit"]
    fiten = by["english", "fit"]
    for lang, fitpart in (("italian", fitit), ("english", fiten)):
        all_pred[lang]["constant"] = np.full(
            len(keys[lang]), fitpart["w"] @ fitpart["y"]
        )
    if (
        digest(HIST / "heldout-predictions.csv")
        != read(HIST / "prediction-lock.json")["forecast_sha256"]
    ):
        raise ValueError("Earlier English comparator forecasts changed")
    hist = {r["id"]: r for r in csv_rows(HIST / "heldout-predictions.csv", ",")}
    for k in ("integration", "lexical_surprisal"):
        all_pred["english"]["previous_" + k] = np.array(
            [float(hist[i][k]) for i in keys["english"]]
        )
    study = {r["id"]: r["source"] for r in read(HIST / "stimuli.json")["material"]}
    all_pred["english"]["study"] = np.array([study[k] for k in keys["english"]])
    folder = OUT / ("form-preflight" if form_only else "forecast")
    folder.mkdir(exist_ok=True)
    output = folder / "predictions.npz"
    np.savez_compressed(
        output,
        **{
            lang + "." + k: v
            for lang, d in all_pred.items()
            for k, v in {"keys": keys[lang], **d}.items()
        },
    )
    weights = folder / "weights.npz"
    np.savez_compressed(weights, **params)
    effects = {
        "fit": audit,
        "native": not form_only,
        "official_affine": {"intercept": base, "slope": slope},
        "source_train_and_dev_only": True,
        "new_test_outcomes_opened": False,
        "official_unscorable": note["undefined_adjacent_sentence_pairs"],
        "predictions": digest(output),
        "parameters": digest(weights),
        "feature_file": digest(feature_path),
        "protocol": digest(PLAN),
        "code": {
            p.name: digest(p)
            for p in Path(__file__).parent.glob("coherence_order_*.py")
        },
    }
    (folder / "seal.json").write_text(json.dumps(effects, indent=2) + "\n")
    print({k: v for k, v in audit.items() if k.endswith(".mixture")}, digest(output))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--form-only", action="store_true")
    run(p.parse_args().form_only)
