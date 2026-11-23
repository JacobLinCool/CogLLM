"""Freeze independent reading forecasts on all summaries, and calibrate separately using only source stories."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_data import OUT as EARLY, read
from research.reading_validation.coherence_order_estimator import (
    equal_group_weights,
    fit,
    predict,
    prepare,
)
from research.reading_validation.coherence_order_fit import source_targets
from research.reading_validation.danish_coherence_forecast import transported
from research.reading_validation.italian_contrast_data import OUT as AUG
from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.summeval_data import OUT, PLAN, RAW, WORK
from research.reading_validation.summeval_estimator import third_task
from research.reading_validation.transport_word_archive import article_boundaries


def judgments(role):
    if role not in ("calibration", "confirm"):
        raise ValueError("Future articles retain protected judgments")
    geo = read(OUT / "story-geometry.json")
    out = {}
    for v in rows(RAW):
        if geo[v["id"]]["role"] != role:
            continue
        experts = [d["coherence"] for d in v["expert_annotations"]]
        crowd = [d["coherence"] for d in v["turker_annotations"]]
        if (
            len(experts) != 3
            or len(crowd) != 5
            or any(n not in [1, 2, 3, 4, 5] for n in [*experts, *crowd])
        ):
            raise ValueError("Unexpected original human coherence scale")
        out[v["id"], v["model_id"]] = (experts, crowd)
    return out


def archive_data(archive):
    rec = read(archive / "receipt.json")
    info = read(OUT / "inputs.json")
    geo = read(OUT / "story-geometry.json")
    src = {r["id"]: r["text"] for r in rows(WORK / "local-stimuli.jsonl")}
    ctrl = read(OUT / "control-forecasts.json")
    dest = defaultdict(list)
    if (
        digest(archive / "features.npz") != rec["file"]
        or rec["source"] != info["packet"]
        or digest(PLAN) != info["protocol"]
        or digest(OUT / "control-forecasts.json")
        != read(OUT / "control-seal.json")["forecasts"]
    ):
        raise ValueError("Unregistered summary candidates or reference")
    for k, r in geo.items():
        for m, doc in r["systems"].items():
            dest[doc].append((k, m))
    if set(src) != set(rec["articles"]) | set(rec["skipped"]):
        raise ValueError("Native capture does not enumerate every news output")
    ids = []
    x = []
    marks = {}
    omitted = dict(rec["skipped"])
    aux = {}
    with np.load(archive / "features.npz", allow_pickle=False) as w:
        for doc, details in sorted(rec["articles"].items()):
            items = dest[doc]
            stories = {k for k, _ in items}
            if len(stories) != 1:
                raise ValueError("A unique decoded draft appears in two source stories")
            story = next(iter(stories))
            t = src[doc]
            b, _ = article_boundaries(w, doc, details, t)
            if not b:
                omitted[doc] = "no_certified_boundary"
                continue
            for (lo, hi), v in b.items():
                ids.append(
                    (f"{doc}/{lo}/{hi}", doc, story, "news", geo[story]["role"], doc)
                )
                x.append(v)
            eligible = details["essay"]
            marks[doc] = (
                [
                    float(np.mean([1 - r["essay_hazard"] for r in eligible])),
                    float(np.mean([1 - r["essay_lexical"] for r in eligible])),
                ]
                if eligible
                else [float("nan")] * 2
            )
            valid = w[doc + ".char"][:, 4].astype(bool)
            s = float(np.mean(w[doc + ".surprisal"][valid]))
            ds = [ctrl[k + "/" + m] for k, m in items]
            forms = np.mean([e["form"] for e in ds], axis=0)
            extra = np.mean([e["reference_overlap"] for e in ds], axis=0)
            g = float(np.mean([e["source_aware_gpt4"] for e in ds]))
            aux[doc] = {
                "form": list(forms) + [s],
                "refs": extra,
                "geval": g,
                "n_stops": len(eligible),
                "story": story,
                "systems": [m for _, m in items],
            }
    return np.array(ids), np.asarray(x), marks, aux, omitted, rec


def open_old():
    wpath = EARLY / "forecast/weights.npz"
    new = AUG / "weights.npz"
    if (
        digest(wpath) != read(EARLY / "forecast/seal.json")["parameters"]
        or digest(new) != read(AUG / "prediction-seal.json")["parameters"]
    ):
        raise ValueError("Prior reader weights changed")
    with (
        np.load(wpath, allow_pickle=False) as w,
        np.load(new, allow_pickle=False) as extra,
    ):
        p = dict(w)
        updated = {**p, **dict(extra)}
    return p, updated


def source_blocks(p):
    source = source_targets()
    blocks = []
    with np.load(EARLY / "native-features.npz", allow_pickle=False) as w:
        ids = w["ids"]
        x = w["x"]
        for lang in ("english", "italian"):
            ix = (ids[:, 4] == "fit") & (
                ids[:, 3] == ("cohesentia" if lang == "english" else "italian")
            )
            z = (x[ix] - p[lang + ".center"]) / p[lang + ".scale"]
            blocks.append(prepare(ids[ix], z, lang, source[lang]))
    return blocks


def ridge(features, y, groups):
    ws = equal_group_weights(groups)
    mean = ws @ features
    scale = np.sqrt(ws @ ((features - mean) ** 2))
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (features - mean) / scale
    avg = ws @ y
    coef = np.linalg.solve(
        z.T @ (z * ws[:, None]) + 0.01 * np.eye(z.shape[1]), z.T @ (ws * (y - avg))
    )
    return lambda v: avg + ((v - mean) / scale) @ coef


def run(archive):
    ids, x, mark, aux, omitted, rec = archive_data(archive)
    p, aug = open_old()
    orig = transported(ids, x, p)
    new = transported(ids, x, aug)
    keys = orig.pop("keys")
    if not np.array_equal(keys, new["keys"]):
        raise ValueError("Native passage predictions shifted")
    m = np.array([mark[k] for k in keys])
    o = {
        **orig,
        "order_augmented": new["adaptive"],
        "essay_hazard": m[:, 0],
        "essay_lexical": m[:, 1],
        "geval": np.array([aux[k]["geval"] for k in keys]),
    }
    groups = np.array([aux[k]["story"] for k in keys])
    geo = read(OUT / "story-geometry.json")
    roles = np.array([geo[g]["role"] for g in groups])
    nf = np.array([aux[k]["n_stops"] for k in keys])
    frozen = OUT / "frozen-predictions.npz"
    np.savez_compressed(
        frozen, keys=keys, stories=groups, roles=roles, eligible_essay_stops=nf, **o
    )
    lock = {
        "frozen_forecasts": digest(frozen),
        "originals": rec["file"],
        "roles": dict(zip(*np.unique(roles, return_counts=True), strict=True)),
        "before_any_confirm_or_future_ballot": True,
        "missing_texts": omitted,
    }
    (OUT / "frozen-seal.json").write_text(
        json.dumps(lock, indent=2, default=int) + "\n"
    )
    target = judgments("calibration")
    train = roles == "calibration"
    truth = {}
    for k, g in zip(keys[train], groups[train], strict=True):
        truth[k] = float(np.mean([target[g, sys][0] for sys in aux[k]["systems"]]))
    forms = np.array([aux[k]["form"] for k in keys])
    ref = np.column_stack([forms, np.array([aux[k]["refs"] for k in keys])])
    mult = np.column_stack([forms, o["adaptive"], m])
    with_hazard = np.isfinite(m).all(1)
    cal = {}
    for n, features, mask in [
        ("form_calibrated", forms, np.ones(len(keys), bool)),
        ("references_calibrated", ref, np.ones(len(keys), bool)),
        (
            "frozen_plus_form",
            np.column_stack([forms, o["adaptive"]]),
            np.ones(len(keys), bool),
        ),
        ("both_heads_plus_form", mult, with_hazard),
        ("matched_form", forms, with_hazard),
    ]:
        sel = train & mask
        ys = np.asarray([truth[k] for k in keys[sel]])
        cal[n] = np.full(len(keys), np.nan)
        cal[n][mask] = ridge(features[sel], ys, groups[sel])(features[mask])
    systems = defaultdict(list)
    for (g, sys), (v, _) in target.items():
        systems[sys].append(float(np.mean(v)))
    cal["system_mean"] = np.array(
        [np.mean([np.mean(systems[sys]) for sys in aux[k]["systems"]]) for k in keys]
    )
    z = (x - p["english.center"]) / p["english.scale"]
    tr = ids[:, 4] == "calibration"
    d = prepare(ids[tr], z[tr], "italian", truth)
    fall = prepare(ids, z, "italian")
    a, diag_one = fit([d])
    j, diag = third_task(source_blocks(p), p["representation.shared"], d)
    if not np.array_equal(keys, fall["keys"]):
        raise ValueError("Native third-task article grouping changed")
    cal["independent_native"] = predict(a, fall)
    cal["shared_native"] = predict(j, fall)
    weights = OUT / "new-native-weights.npz"
    np.savez_compressed(
        weights,
        independent=a,
        shared_third_task=j,
        shared_all_parameters=diag.pop("full_parameters"),
    )
    path = OUT / "calibrated-predictions.npz"
    np.savez_compressed(path, keys=keys, **cal)
    code = {
        name: digest(Path(__file__).with_name(name))
        for name in (
            "summeval_data.py",
            "summeval_controls.py",
            "summeval_estimator.py",
            "summeval_forecast.py",
            "summeval_score.py",
            "summeval-protocol.md",
        )
    }
    info = {
        "code": code,
        "frozen": digest(frozen),
        "calibrated": digest(path),
        "weights": digest(weights),
        "third_task_optimization": diag,
        "independent_optimization": diag_one,
        "calibrated_drafts": int(train.sum()),
        "calibration_source_articles": len(set(groups[train])),
        "confirmed_or_future_scores_read": False,
    }
    (OUT / "prediction-seal.json").write_text(json.dumps(info, indent=2) + "\n")
    print(info["third_task_optimization"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    run(parser.parse_args().archive)
