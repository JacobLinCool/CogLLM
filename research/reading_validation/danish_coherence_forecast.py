"""Apply the sealed Italian and shared readouts to one-pass Danish article features."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_data import OUT as FIT
from research.reading_validation.coherence_order_data import read
from research.reading_validation.coherence_order_estimator import predict, prepare
from research.reading_validation.coherence_order_features import (
    BASE_COLUMNS,
    PRED_COLUMNS,
)
from research.reading_validation.danish_coherence_data import OUT, PLAN, WORK, entries
from research.reading_validation.local_forecast import digest
from research.reading_validation.memory_data import h
from research.reading_validation.transport_word_archive import article_boundaries


def vectorize(archive):
    r = read(archive / "receipt.json")
    inside = read(OUT / "inputs.json")
    if (
        r["file"] != digest(archive / "features.npz")
        or r["source"] != inside["packet"]
        or inside["packet"] != digest(WORK / "stimuli.jsonl")
    ):
        raise ValueError("Unexpected Danish original evidence")
    all_docs = {r["id"]: r for r in entries("test")}
    missing = Counter(r["skipped"].values())
    geo = read(OUT / "geometry.json")
    ids = []
    features = []
    incomplete = {}
    if set(all_docs) != (set(r["skipped"]) | set(r["articles"])) or set(
        r["skipped"]
    ) & set(r["articles"]):
        raise ValueError(
            "Native receipt fails to enumerate the Danish original inventory"
        )
    with np.load(archive / "features.npz", allow_pickle=False) as z:
        for key, doc in sorted(r["articles"].items()):
            t = all_docs[key]["text"]
            boundary, total = article_boundaries(z, key, doc, t)
            for (a, b), row in boundary.items():
                ids.append(
                    (h(f"{key}/{a}/{b}"), key, geo[key]["group"], "danish", "held", key)
                )
                features.append(row)
            retained = len(boundary)
            if retained < total:
                incomplete[key] = [retained, total]
            if not retained:
                missing["no_scorable_boundary"] += 1
    return (
        np.asarray(ids),
        np.asarray(features),
        {
            "archive": r["file"],
            "worker_seconds": r["seconds"],
            "native_articles": len(r["articles"]),
            "encoder_batches": r["encoder_batches"],
            "prefix_checks": r["checks"],
            "incomplete_boundaries": incomplete,
            "omitted_articles": dict(missing),
        },
    )


def transported(ids, x, params):
    out = {}
    z = (x - params["italian.center"]) / params["italian.scale"]
    for arm, size in (
        ("form", len(BASE_COLUMNS)),
        ("surprisal", len(BASE_COLUMNS) + len(PRED_COLUMNS)),
        ("representation", x.shape[1]),
    ):
        group = prepare(ids, z[:, :size], "italian")
        if "keys" in out and not np.array_equal(out["keys"], group["keys"]):
            raise ValueError("Mismatched article estimates")
        out["keys"] = group["keys"]
        out[arm] = predict(params[arm + ".italian"], group)
    out["shared"] = predict(params["representation.shared"], group)
    a = params["italian.mixture"].item()
    out["adaptive"] = (1 - a) * out["representation"] + a * out["shared"]
    return out


def run(archive):
    source = FIT / "forecast/weights.npz"
    seal = read(FIT / "forecast/seal.json")
    if (
        not seal["source_train_and_dev_only"]
        or digest(source) != seal["parameters"]
        or read(OUT / "inputs.json")["protocol"] != digest(PLAN)
    ):
        raise ValueError("Training assumptions do not match the transfer protocol")
    ids, x, note = vectorize(archive)
    with np.load(source, allow_pickle=False) as params:
        out = transported(ids, x, params)
        # Confirm that the same inference function exactly reproduces every original Italian forecast.
        with np.load(FIT / "native-features.npz", allow_pickle=False) as q:
            ix = (q["ids"][:, 3] == "italian") & (q["ids"][:, 4] == "held")
            old = transported(q["ids"][ix], q["x"][ix], params)
        with np.load(
            FIT / "forecast/predictions.npz", allow_pickle=False
        ) as historical:
            if not np.array_equal(old["keys"], historical["italian.keys"]):
                raise ValueError("Source-held material identity differs")
            err = max(
                float(np.max(abs(v - historical["italian." + k])))
                for k, v in old.items()
                if k != "keys"
            )
        if err > 1e-9:
            raise ArithmeticError(
                "Danish code cannot reproduce authentic Italian predictions"
            )
    ctrl = OUT / "local-controls.npz"
    if digest(ctrl) != read(OUT / "control-seal.json")["control"]:
        raise ValueError("Unexpected Danish target-supervised control")
    with np.load(ctrl, allow_pickle=False) as q:
        order = {k: i for i, k in enumerate(q["keys"])}
        idx = [order[k] for k in out["keys"]]
        for k in ("danish_form", "danish_lexical"):
            out[k] = q[k][idx]
    geo = read(OUT / "geometry.json")
    out["group"] = np.array([geo[k]["group"] for k in out["keys"]])
    out["domain"] = np.array([geo[k]["domain"] for k in out["keys"]])
    out["domain_only"] = np.ones(len(idx)) * 2.0
    dst = OUT / "forecasts.npz"
    np.savez_compressed(dst, **out)
    note.update(
        {
            "predictions": digest(dst),
            "geometry": digest(OUT / "geometry.json"),
            "source_model": digest(source),
            "source_splits": digest(FIT / "forecast/seal.json"),
            "prior_test_grade_used": False,
            "source_reproduction_max_abs": err,
            "forecasted_passages": len(idx),
            "target_domain_counts": dict(Counter(out["domain"])),
            "protocol": digest(PLAN),
            "science_code": {
                p.name: digest(p)
                for p in (
                    *Path(__file__).parent.glob("danish_coherence_*.py"),
                    Path(__file__).with_name("transport_word_archive.py"),
                )
            },
        }
    )
    (OUT / "prediction-seal.json").write_text(json.dumps(note, indent=2) + "\n")
    print(
        {
            k: note[k]
            for k in (
                "forecasted_passages",
                "source_reproduction_max_abs",
                "predictions",
            )
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    run(p.parse_args().archive)
