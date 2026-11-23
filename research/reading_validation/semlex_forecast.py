"""Calibrate on an earlier study, then freeze external two-candidate forecasts without grades."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
from scipy.special import softmax

from research.reading_validation.local_forecast import WORD, digest, rows
from research.reading_validation.nns_fit import all_features
from research.reading_validation.nns_inference import WordReport
from research.reading_validation.nns_rewrite_features import comparisons
from research.reading_validation.nns_rewrite_fit import multiclass_mix, ternary, weights
from research.reading_validation.nns_rewrite_material import (
    DATA as TRAIN_CACHE,
    OUT as TRAIN,
    focus,
)
from research.reading_validation.semlex_l4 import PACKET, KERNEL
from research.reading_validation.semlex_material import DATA, OUT

OLD_REFERENCES = {
    "paper_frequency": "frequency.txt",
    "paper_adaptive_ngram": "adaptive-ngram.txt",
    "paper_psycholinguistic": "psycholing.txt",
}
NATIVE = (
    "joint_native_mix",
    "lexical_plus_surprisal",
    "shared_contextual",
    "learned_lexical",
    "raw_target_surprisal",
    "word_frequency",
)
METHODS = (*NATIVE, *OLD_REFERENCES)


def source_fit():
    seal = json.loads((TRAIN / "forecast-seal.json").read_bytes())
    with np.load(TRAIN_CACHE / "word-features.npz") as f:
        bank = dict(f)
    actual = comparisons(bank, "held", outcomes=True)
    if (
        digest(TRAIN / "forecasts.npz") != seal["forecasts_sha256"]
        or digest(TRAIN / "weights.npz") != seal["weights_sha256"]
    ):
        raise ValueError("Earlier source weights/forecasts changed")
    with np.load(TRAIN / "forecasts.npz", allow_pickle=False) as f:
        if not np.array_equal(f["r_keys"], actual["keys"]):
            raise ValueError("Source pairs changed")
        a = f["reference.learned"]
        b = f["r.contextual.adaptive"]
        seen = f["reference.common"]
    k = actual["counts"][seen]
    y = k / k.sum(1, keepdims=True)
    w = weights(actual["doc"][seen], actual["family"][seen])
    h = multiclass_mix(a[seen], b[seen], y, w)
    val = lambda p: float(-w @ np.sum(y * np.log(p), axis=1))
    summary = {
        "source_families": len(set(actual["family"][seen])),
        "source_pairs": len(k),
        "votes": int(k.sum()),
        "source_forecasts_sha256": seal["forecasts_sha256"],
        "source_weights_sha256": seal["weights_sha256"],
        "convex_weight_on_shared_contextual": h,
        "source_crossentropy": {
            "reference": val(a[seen]),
            "shared": val(b[seen]),
            "mixture": val((1 - h) * a[seen] + h * b[seen]),
        },
        "external_gold_accessed": False,
        "definition_sha256": digest(Path(__file__)),
    }
    (OUT / "source-calibration.json").write_text(json.dumps(summary, indent=2) + "\n")


def read_ranks(path):
    ranks = {}
    for row, line in enumerate(path.read_text().splitlines(), 301):
        head, sep, groups = line.partition(" rankings: ")
        if not sep or head != f"Sentence {row - 300}":
            raise ValueError("Unexpected organizer ranking identifier")
        chunks = re.findall(r"\{([^{}]*)\}", groups)
        if not chunks or re.sub(r"\{[^{}]*\}", "", groups).strip():
            raise ValueError("Unexpected organizer ranking syntax")
        lookup = {}
        ambiguous = set()
        for i, g in enumerate(chunks):
            for word in g.split(","):
                w = word.strip()
                if w in lookup and lookup[w] != i:
                    ambiguous.add(w)
                lookup[w] = i
        ranks[str(row)] = {w: order for w, order in lookup.items() if w not in ambiguous}
    if len(ranks) != 1710:
        raise ValueError("Missing organizer rankings")
    return ranks


def external_bank(extracted):
    rec = json.loads((extracted / "extraction.json").read_bytes())
    if (
        rec["packet_sha256"] != PACKET
        or rec["worker_sha256"] != KERNEL
        or rec["registered_external_wrapper_sha256"]
        != digest(Path(__file__).with_name("semlex_l4.py"))
        or any(digest(extracted / k) != v for k, v in rec["files_sha256"].items())
    ):
        raise ValueError("External public word variants or model receipt changed")
    model = WordReport.load()
    byid = {
        k: i for i, k in enumerate(json.loads((extracted / "ids.json").read_bytes()))
    }
    with np.load(extracted / "signals.npz") as f:
        z = dict(f)
    design = []
    local_ids = []
    surp = []
    meas = []
    for r in rows(DATA / "stimuli.jsonl"):
        text = r["text"]
        wid = r["id"]
        words = np.array([m.span() for m in WORD.finditer(text)], int)
        n = len(words)
        wanted = {tuple(v) for v in r["spans"]}
        ok = np.zeros(n, bool)
        c = np.zeros(n, np.float32)
        p = np.zeros((n, 32), np.float32)
        x = np.zeros((n, 64), np.float16)
        kept = []
        for j, (a, b) in enumerate(words):
            if (a, b) not in wanted:
                continue
            key = focus(wid, a, b)
            i = byid[key]
            if z["native"][i] < 0:
                continue
            ok[j] = True
            c[j] = z["lexical"][i]
            p[j] = z["plaba"][i]
            x[j] = z["context"][i]
            kept.append((key, i))
        if not any(ok):
            continue
        _, base, rates = model._word_design(text, words, ok, c, p, x)
        idx = np.column_stack([np.arange(len(rates)), np.zeros(len(rates), int)])
        design.append(
            tuple(
                all_features(base, idx, rates, a, False)
                for a in ("lexical", "contextual")
            )
        )
        local_ids.extend(k for k, _ in kept)
        surp.extend(z["surprisal"][i] for _, i in kept)
        meas.extend(z["measured_surprisal"][i] for _, i in kept)
    if len(local_ids) != len(set(local_ids)):
        raise ValueError("Multiple native values for one word")
    return (
        {k: i for i, k in enumerate(local_ids)},
        [np.concatenate(a) for a in zip(*design, strict=True)],
        np.array(surp),
        np.array(meas),
        rec,
    )


def source_predictions(x, sp, pairs, weights_, fraction):
    a, b = pairs.T
    outputs = {}
    for j, arm in enumerate(("lexical", "contextual")):
        dif = (x[j][b] - x[j][a]) / (2 * weights_[arm + ".scale"])
        pp = []
        for branch in ("independent", "shared"):
            beta = weights_[arm + "." + branch + ".theta"]
            p = dif.shape[1]
            pp.append(softmax(ternary(dif @ beta[p : 2 * p], beta[-1]), axis=1))
        h = float(weights_[arm + ".r.fraction"])
        outputs[arm] = (1 - h) * pp[0] + h * pp[1]
    full = np.column_stack(
        [x[0], np.log1p(sp[:, 0]), np.log1p(np.maximum(0, sp[:, 1] - sp[:, 0]))]
    )
    th = weights_["reference.theta"]
    B = softmax(
        ternary(
            (full[b] - full[a]) / (2 * weights_["reference.scale"]) @ th[:-1], th[-1]
        ),
        axis=1,
    )
    C = outputs["contextual"]
    mix = (1 - fraction) * B + fraction * C
    return np.column_stack(
        [q[:, 0] - q[:, 1] for q in (mix, B, C, outputs["lexical"])]
        + [sp[b, 0] - sp[a, 0], x[0][a, 2] - x[0][b, 2]]
    )


def run(extracted):
    material = json.loads((OUT / "material.json").read_bytes())
    if digest(DATA / "geometry.json") != material["geometry_sha256"]:
        raise ValueError("Different context geometry")
    original = json.loads((DATA / "geometry.json").read_bytes())
    source = json.loads((OUT / "source-calibration.json").read_bytes())
    if source["definition_sha256"] != digest(Path(__file__)) or source[
        "source_weights_sha256"
    ] != digest(TRAIN / "weights.npz"):
        raise ValueError("Source calibrator or parameters changed")
    alpha = source["convex_weight_on_shared_contextual"]
    ids, x, sp, meas, rec = external_bank(extracted)
    historical = {k: read_ranks(DATA / path) for k, path in OLD_REFERENCES.items()}
    keys = []
    pairs = []
    paper = []
    incomplete = 0
    for key, item in sorted(original.items(), key=lambda e: int(e[0])):
        eligible = item["variants"]
        for a, b in combinations(sorted(eligible), 2):
            ia = ids.get(eligible[a]["word"])
            ib = ids.get(eligible[b]["word"])
            if ia is None or ib is None or not (meas[ia].all() and meas[ib].all()):
                incomplete += 1
                continue
            try:
                rank = [
                    np.sign(table[key][b] - table[key][a])
                    for table in historical.values()
                ]
            except KeyError:
                incomplete += 1
                continue
            keys.append((key, item["lemma"], a, b))
            pairs.append((ia, ib))
            paper.append(rank)
    ij = np.asarray(pairs)
    with np.load(TRAIN / "weights.npz", allow_pickle=False) as f:
        cols = source_predictions(x, sp, ij, f, alpha)
    matrix = np.column_stack([cols, np.array(paper)])
    if not np.isfinite(matrix).all():
        raise ValueError("Incomplete candidate estimate")
    forecast = OUT / "forecasts.npz"
    np.savez_compressed(
        forecast,
        keys=np.asarray(keys),
        signed_a_preference=matrix,
        methods=np.array(METHODS),
    )
    receipt = {
        "external_outcomes_seen": False,
        "forecast_sha256": digest(forecast),
        "pairs": len(keys),
        "incomplete_pairs": incomplete,
        "source_calibration_sha256": digest(OUT / "source-calibration.json"),
        "material_sha256": digest(OUT / "material.json"),
        "reading_definition_sha256": digest(Path(__file__)),
        "l4": rec,
    }
    (OUT / "forecasts-seal.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("External frozen pairs", len(keys), "omitted", incomplete, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--source-fit", action="store_true")
    choice.add_argument("--extraction", type=Path)
    args = parser.parse_args()
    source_fit() if args.source_fit else run(args.extraction)
