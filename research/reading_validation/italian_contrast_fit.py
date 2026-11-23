"""Fit and seal the source-order auxiliary model separately from the original comparison."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_data import ITALIAN, read, texts
from research.reading_validation.coherence_order_data import OUT as FIT
from research.reading_validation.coherence_order_estimator import (
    mixture,
    predict,
    prepare,
)
from research.reading_validation.coherence_order_fit import source_targets
from research.reading_validation.danish_coherence_forecast import transported, vectorize
from research.reading_validation.italian_contrast_data import OUT, PLAN
from research.reading_validation.italian_contrast_estimator import fit, layout
from research.reading_validation.italian_order_data import OUT as H
from research.reading_validation.italian_order_forecast import read_banks
from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.memory_data import h
from research.reading_validation.transport_word_archive import article_boundaries


def source_pairs(archive, ids, x, params):
    samples = read(OUT / "source-geometry.json")
    s = read(OUT / "inputs.json")
    r = read(archive / "receipt.json")
    path = ITALIAN / "source-counterfactuals.jsonl"
    if (
        digest(path) != s["source"]
        or r["source"] != s["source"]
        or digest(archive / "features.npz") != r["file"]
        or digest(OUT / "source-geometry.json") != s["geometry"]
    ):
        raise ValueError("Source intervention archive mismatch")
    source = {v["id"]: v["text"] for v in rows(path)}
    available = defaultdict(dict)
    needed = {i["observed"] for i in samples.values() if i["eligible"]}
    center = params["italian.center"]
    scale = params["italian.scale"]
    for idx, row in zip(ids, x, strict=True):
        if idx[1] in needed:
            available[idx[1]][idx[0]] = (row - center) / scale
    with np.load(archive / "features.npz", allow_pickle=False) as f:
        for k, p in r["articles"].items():
            ss, _ = article_boundaries(f, k, p, source[k])
            available[k] = {
                h(f"{k}/{a}/{b}"): (v - center) / scale for (a, b), v in ss.items()
            }
    splits = {"fit": [], "dev": []}
    omitted = Counter()
    for key, rec in sorted(samples.items()):
        if not rec["eligible"]:
            continue
        c = rec["canonical"]
        t = rec["permuted"]
        ca = available[c]
        sw = available[t]
        if not ca or not sw:
            omitted[rec["role"]] += 1
            continue
        a, b = rec["canonical_interval"]
        ac = h(f"{c}/{a}/{b}")
        a, b = rec["permuted_interval"]
        bc = h(f"{t}/{a}/{b}")
        local = (
            (sw[bc] - ca[ac])
            if rec["canonical_has_predecessor"] and ac in ca and bc in sw
            else None
        )
        splits[rec["role"]].append(
            {
                "group": rec["source"],
                "id": key,
                "canonical": np.array(list(ca.values())),
                "permuted": np.array(list(sw.values())),
                "local": local,
            }
        )
    return splits, dict(omitted)


def run(source, original, danish, permuted):
    model = FIT / "forecast"
    seal = read(model / "seal.json")
    p = model / "weights.npz"
    if (
        digest(p) != seal["parameters"]
        or digest(FIT / "native-features.npz") != seal["feature_file"]
        or digest(PLAN) != read(OUT / "inputs.json")["plan"]
    ):
        raise ValueError("Native training model does not match source registration")
    with np.load(p, allow_pickle=False) as f:
        params = dict(f)
    with np.load(FIT / "native-features.npz", allow_pickle=False) as f:
        ids = f["ids"]
        x = f["x"]
    pairs, skips = source_pairs(source, ids, x, params)
    structure = layout(pairs["fit"])
    target = source_targets()
    blocks = {}
    for language in ("english", "italian"):
        z = (x - params[language + ".center"]) / params[language + ".scale"]
        mask = (
            (ids[:, 3] == "italian")
            if language == "italian"
            else (ids[:, 3] != "italian")
        )
        for role in ("fit", "dev", "held"):
            sel = mask & (ids[:, 4] == role)
            blocks[language, role] = prepare(
                ids[sel], z[sel], language, None if role == "held" else target[language]
            )
    learned, checks = fit(
        [blocks["english", "fit"], blocks["italian", "fit"]],
        structure,
        params["representation.shared"],
        seal["fit"]["representation.shared"],
    )
    updated = dict(params)
    updated["representation.shared"] = learned
    outputs = {}
    for lang in ("english", "italian"):
        d = blocks[lang, "dev"]
        ind = params["representation." + lang]
        a = mixture(d, predict(ind, d), predict(learned, d))
        updated[lang + ".mixture"] = np.array(a)
        held = blocks[lang, "held"]
        outputs[lang + ".keys"] = held["keys"]
        i = predict(ind, held)
        j = predict(learned, held)
        outputs[lang + ".augmented_shared"] = j
        outputs[lang + ".augmented_adaptive"] = (1 - a) * i + a * j
    ix, dx, _ = vectorize(danish)
    pred = transported(ix, dx, updated)
    outputs["danish.keys"] = pred["keys"]
    outputs["danish.augmented_adaptive"] = pred["adaptive"]
    outputs["danish.augmented_shared"] = pred["shared"]
    des = read(H / "order-pairs.json")
    subjects = {
        k: rec["source"]
        for rec in des.values()
        for k in (rec["canonical"], rec["permuted"])
    }
    alt = {r["id"]: r["text"] for r in rows(ITALIAN / "counterfactuals.jsonl")}
    archives = [
        (
            original,
            read(
                Path("research/results/reading-validation-state-transport/inputs.json")
            )["source"],
            texts(),
        ),
        (permuted, read(H / "inputs.json")["source"], alt),
    ]
    pp, br, absent, sources = read_banks(archives, subjects, updated)
    results = []
    for k, rec in des.items():
        ca = rec["canonical"]
        sw = rec["permuted"]
        if ca not in pp or sw not in pp:
            continue
        left = br.get((ca, *rec["canonical_interval"]))
        right = br.get((sw, *rec["permuted_interval"]))
        results.append(
            {
                "id": k,
                "group": rec["source"],
                "canonical_has_predecessor": rec["canonical_has_predecessor"],
                "observed_type": rec["observed_type"],
                "canonical_minus_permuted_passage": {
                    n: pp[ca][n] - pp[sw][n] for n in ("adaptive", "shared")
                },
                "permuted_minus_canonical_break_probability": None
                if left is None or right is None
                else {n: right[n] - left[n] for n in ("adaptive", "shared")},
            }
        )
    rp = OUT / "order-differences.json"
    rp.write_text(json.dumps(results, indent=2) + "\n")
    dest = OUT / "predictions.npz"
    np.savez_compressed(dest, **outputs)
    mod = OUT / "weights.npz"
    np.savez_compressed(
        mod,
        **{
            k: updated[k]
            for k in ("representation.shared", "english.mixture", "italian.mixture")
        },
    )
    notes = {
        "algorithm": checks,
        "source": {k: v for k, v in structure.items() if isinstance(v, int)},
        "source_missing": skips,
        "source_separation": read(OUT / "inputs.json"),
        "source_fit_observed_type": dict(
            Counter(
                read(OUT / "source-geometry.json")[e["id"]]["observed_type"]
                for e in pairs["fit"]
            )
        ),
        "development_structural_count": len(pairs["dev"]),
        "mixture": {k: float(updated[k + ".mixture"]) for k in ("english", "italian")},
        "predictions": digest(dest),
        "paired_structural_predictions": digest(rp),
        "parameters": digest(mod),
        "native_source": digest(source / "receipt.json"),
        "native_danish": digest(danish / "receipt.json"),
        "native_original_and_variant": sources,
        "variant_absent": absent,
        "prior_source_model": digest(p),
        "study_implementation": {
            a.name: digest(a)
            for a in Path(__file__).parent.glob("italian_contrast_*.py")
        },
        "new_test_outcomes_used": False,
    }
    (OUT / "prediction-seal.json").write_text(json.dumps(notes, indent=2) + "\n")
    print(checks, notes["source"], notes["mixture"])


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "original", "danish", "permuted"):
        a.add_argument(k, type=Path)
    run(**vars(a.parse_args()))
